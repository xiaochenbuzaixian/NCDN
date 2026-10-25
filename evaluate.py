import os
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

class EEGNet(nn.Module):
    """标准 EEGNet 分类器. 输入: (B, 30, 4000)"""
    def __init__(self, n_channels=30, n_samples=4000, n_classes=2, F1=8, D=2, F2=16, downsample=8):
        super().__init__()
        self.downsample = downsample
        n_in = n_samples // downsample
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        feat_len = n_in // 32
        self._feat_dim = F2 * feat_len
        self.fc = nn.Linear(self._feat_dim, n_classes)

    def extract_features(self, x):
        if self.downsample > 1:
            x = x[:, :, ::self.downsample]
        x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        return self.separable(x).flatten(1)

    def forward(self, x):
        return self.fc(self.extract_features(x))

class PhysiofNIRSBranch(nn.Module):
    """
    生理特征增强型 fNIRS 分支:
    同时提取任务期 [0s~20s] (采样点 50:250) 的:
    1. 4个时间窗均值 (0-5s, 5-10s, 10-15s, 15-20s)
    2. 早期上升斜率与主峰斜率
    3. 左右半球对称通道差分特征 (Hemispheric Lateralization)
    4. 1D 时空卷积特征
    """
    def __init__(self, n_channels=36, out_dim=32):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(n_channels, 24, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(24),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten()
        )
        # 统计特征维度: 4窗口均值(36*4) + 2斜率(36*2) + 左右半球差分(18*2) = 252
        stat_dim = 36 * 6 + 18 * 2
        self.proj = nn.Sequential(
            nn.BatchNorm1d(24 * 4 + stat_dim),
            nn.Dropout(0.3),
            nn.Linear(24 * 4 + stat_dim, out_dim),
            nn.ELU()
        )
        self.aux_fc = nn.Linear(out_dim, 2)

    def forward(self, f):
        # f: (B, 36, 256), 取任务期 50:250
        w1 = f[:, :, 50:100].mean(dim=-1)
        w2 = f[:, :, 100:150].mean(dim=-1)
        w3 = f[:, :, 150:200].mean(dim=-1)
        w4 = f[:, :, 200:250].mean(dim=-1)
        slope1 = w2 - w1
        slope2 = w4 - w2
        # 左右半球通道偏侧化对比 (前18通道 vs 后18通道)
        lat_early = w2[:, :18] - w2[:, 18:]
        lat_late = w4[:, :18] - w4[:, 18:]

        stats = torch.cat([w1, w2, w3, w4, slope1, slope2, lat_early, lat_late], dim=-1)
        conv_f = self.conv_branch(f[:, :, 50:])
        feat = self.proj(torch.cat([conv_f, stats], dim=-1))
        return feat, self.aux_fc(feat)

class HybridNet(nn.Module):
    """
    防梯度抢占的协同 HybridNet:
    以预训练 EEGNet 为锚点，通过自适应门控融合 fNIRS 生理特征分支
    """
    def __init__(self, pretrained_eegnet: EEGNet):
        super().__init__()
        self.eeg_net = copy.deepcopy(pretrained_eegnet)
        self.fnirs_branch = PhysiofNIRSBranch(n_channels=36, out_dim=32)
        self.fusion_fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.eeg_net._feat_dim + 32, 32),
            nn.ELU(),
            nn.Linear(32, 2)
        )
        self.alpha = nn.Parameter(torch.tensor(0.35))

    def forward(self, eeg, fnirs):
        e_feat = self.eeg_net.extract_features(eeg)
        e_logits = self.eeg_net.fc(e_feat)
        f_feat, f_logits = self.fnirs_branch(fnirs)
        joint_logits = self.fusion_fc(torch.cat([e_feat, f_feat], dim=-1))
        gate = torch.clamp(self.alpha, 0.1, 0.8)
        final_logits = (1.0 - gate) * e_logits + gate * (0.6 * joint_logits + 0.4 * f_logits)
        return final_logits, e_logits, f_logits

def classification_metrics(y_true, y_pred):
    return {
        "ACC": accuracy_score(y_true, y_pred) * 100,
        "PRE": precision_score(y_true, y_pred, pos_label=0, zero_division=0) * 100,
        "SEN": recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100,
        "SPE": recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
    }

def train_eegnet(tr_loader, te_loader, epochs=50, lr=1e-3, device="cuda", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EEGNet(downsample=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for e, y in tr_loader:
            e, y = e.to(device), y.to(device)
            loss = crit(model(e), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for e, y in te_loader:
            logits = model(e.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    return model, classification_metrics(np.concatenate(targets), np.concatenate(preds))

def train_hybridnet(base_eegnet, tr_loader, te_loader, epochs=40, lr=8e-4, device="cuda", seed=42):
    """微调 EEG 分支 + 训练 fNIRS 生理分支 (深度监督损失杜绝梯度抢占)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HybridNet(base_eegnet).to(device)
    # EEG 分支使用较小学习率保持稳定，fNIRS 分支与融合头使用标准学习率
    opt = torch.optim.AdamW([
        {"params": model.eeg_net.parameters(), "lr": lr * 0.15},
        {"params": model.fnirs_branch.parameters(), "lr": lr},
        {"params": model.fusion_fc.parameters(), "lr": lr},
        {"params": [model.alpha], "lr": lr}
    ], weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for e, f, y in tr_loader:
            e, f, y = e.to(device), f.to(device), y.to(device)
            final_logits, e_logits, f_logits = model(e, f)
            loss = crit(final_logits, y) + 0.4 * crit(f_logits, y) + 0.2 * crit(e_logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for e, f, y in te_loader:
            final_logits, _, _ = model(e.to(device), f.to(device))
            preds.append(final_logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    return classification_metrics(np.concatenate(targets), np.concatenate(preds))

def calc_set_metrics(real_arr, synth_arr):
    mse = float(np.mean((real_arr - synth_arr) ** 2))
    rmse = float(np.sqrt(mse))
    corrs = []
    for r, s in zip(real_arr, synth_arr):
        rf, sf = r.ravel() - r.mean(), s.ravel() - s.mean()
        corrs.append(float((rf * sf).sum() / (np.sqrt((rf ** 2).sum() * (sf ** 2).sum()) + 1e-12)))
    curve_pcc, _ = pearsonr(real_arr.mean(axis=(0, 1)), synth_arr.mean(axis=(0, 1)))
    return mse, rmse, float(np.mean(corrs)), float(curve_pcc)

def evaluate_checkpoint(epoch="best", ddim_steps=50):
    with open("configs/shin_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    modality = cfg["data"]["target_modality"]
    processed_dir = cfg["data"]["processed_dir"]
    use_hrf = cfg["data"].get("eeg_hrf", True)

    ckpt_name = f"scdm_{modality.lower()}_best.pt" if str(epoch) == "best" else f"scdm_{modality.lower()}_ep{epoch}.pt"
    ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    topo_dict = torch.load(os.path.join(processed_dir, f"topo_{modality.lower()}.pt"), map_location="cpu")
    topo_ef = topo_dict["topo_ef"].to(device)
    topo_fe = topo_dict["topo_fe"].to(device)
    topo_ee = topo_dict.get("topo_ee", topo_ef).to(device)
    topo_ff = topo_dict.get("topo_ff", topo_fe).to(device)

    unet = SCDMUnet(
        time_dim=cfg["model"]["time_dim"],
        base_channels=cfg["model"].get("base_channels", 128)
    ).to(device)
    model = SCDMDiffusion(
        unet,
        timesteps=cfg["model"]["timesteps"],
        beta_start=cfg["model"]["beta_start"],
        beta_end=cfg["model"]["beta_end"],
        schedule=cfg["model"].get("schedule", "cosine")
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print(f">>> 已加载最优模型权重 ({ckpt_name}, epoch={ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt)
    model.eval()

    eeg_all = torch.load(os.path.join(processed_dir, "eeg_epochs.pt"))
    eeg_cond = torch.load(os.path.join(processed_dir, "eeg_hrf_epochs.pt")) if use_hrf else eeg_all
    fnirs_all = torch.load(os.path.join(processed_dir, f"fnirs_{modality.lower()}_epochs.pt"))
    # 同时加载另一血氧分量 (HbO/HbR) 供真实双模态上限参考
    other_mod = "hbo" if modality.lower() == "hbr" else "hbr"
    fnirs_other = torch.load(os.path.join(processed_dir, f"fnirs_{other_mod}_epochs.pt"))
    labels_all = torch.load(os.path.join(processed_dir, "labels.pt"))

    print(f">>> 正在使用 Topology-Guided DDIM-{ddim_steps} 步采样生成合成 fNIRS...")
    synth_list = []
    bs = cfg["data"]["batch_size"]
    with torch.no_grad():
        for i in tqdm(range(0, len(eeg_cond), bs), desc="DDIM 采样进度"):
            e_batch = eeg_cond[i:i+bs].to(device)
            syn = model.sample(e_batch, topo_ef, topo_fe, topo_ee, topo_ff, steps=ddim_steps).cpu().numpy()
            synth_list.append(syn)

    synth_all = np.concatenate(synth_list, axis=0)
    real_all = fnirs_all.numpy()
    real_dual = 0.6 * real_all - 0.4 * fnirs_other.numpy()  # HbR-HbO 推挽耦合增强信噪比
    eeg_np = eeg_all.numpy()
    labels_np = labels_all.numpy()

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(labels_np), generator=g).numpy()
    val_idx = perm[int(len(labels_np) * cfg["data"]["train_ratio"]):]

    val_mse, val_rmse, val_spcc, val_cpcc = calc_set_metrics(real_all[val_idx], synth_all[val_idx])
    all_mse, all_rmse, all_spcc, all_cpcc = calc_set_metrics(real_all, synth_all)

    print("\n" + "=" * 70)
    print(f"【1. 信号生成质量评估】 (权重: {ckpt_name})")
    print(f"  [独立验证集 N={len(val_idx)}] MSE: {val_mse:.4f} (RMSE: {val_rmse:.4f}) | 单试次 PCC: {val_spcc:.4f} | 均值曲线 PCC: {val_cpcc:.4f}")
    print(f"  [全集口径   N={len(real_all)}] MSE: {all_mse:.4f} (RMSE: {all_rmse:.4f}) | 单试次 PCC: {all_spcc:.4f} | 均值曲线 PCC: {all_cpcc:.4f}")
    print("  *(注: safii-74 报告的 MSE=0.421, PCC=0.437 为全集 N=1740 口径)*")

    # 2. 5 折交叉分类评估
    print("\n>>> 正在运行 5 折 EEGNet / HybridNet 运动想象分类检验...")
    results = {"eeg_only": [], "eeg_real": [], "eeg_synth": []}
    for run in range(5):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=run * 42)
        tr_idx, te_idx = next(sss.split(eeg_np, labels_np))

        e_tr, e_te = torch.from_numpy(eeg_np[tr_idx]), torch.from_numpy(eeg_np[te_idx])
        y_tr, y_te = torch.from_numpy(labels_np[tr_idx]), torch.from_numpy(labels_np[te_idx])
        fr_tr, fr_te = torch.from_numpy(real_dual[tr_idx]), torch.from_numpy(real_dual[te_idx])
        fs_tr, fs_te = torch.from_numpy(synth_all[tr_idx]), torch.from_numpy(synth_all[te_idx])

        # (1) 训练单模态 EEGNet
        trained_eegnet, m_eeg = train_eegnet(
            DataLoader(TensorDataset(e_tr, y_tr), batch_size=32, shuffle=True),
            DataLoader(TensorDataset(e_te, y_te), batch_size=32),
            epochs=50, lr=1e-3, device=device, seed=run * 100
        )
        results["eeg_only"].append(m_eeg)

        # (2) 训练 EEG + Real fNIRS (以当前折 EEGNet 为基底，防止梯度抢占)
        m_real = train_hybridnet(
            trained_eegnet,
            DataLoader(TensorDataset(e_tr, fr_tr, y_tr), batch_size=32, shuffle=True),
            DataLoader(TensorDataset(e_te, fr_te, y_te), batch_size=32),
            epochs=40, lr=8e-4, device=device, seed=run * 100
        )
        results["eeg_real"].append(m_real)

        # (3) 训练 EEG + Synth fNIRS
        m_syn = train_hybridnet(
            trained_eegnet,
            DataLoader(TensorDataset(e_tr, fs_tr, y_tr), batch_size=32, shuffle=True),
            DataLoader(TensorDataset(e_te, fs_te, y_te), batch_size=32),
            epochs=40, lr=8e-4, device=device, seed=run * 100
        )
        results["eeg_synth"].append(m_syn)

    print("\n【2. 下游运动想象 (MI) 分类结果 (5-Run Average)】")
    print(f"{'Mode':<26s} {'ACC':>10s} {'PRE':>10s} {'SEN':>10s} {'SPE':>10s}")
    print("-" * 70)
    for key, title in [
        ("eeg_only", "EEG-only (EEGNet)"),
        ("eeg_real", "EEG + Real fNIRS"),
        ("eeg_synth", "EEG + Synth fNIRS (SCDM)")
    ]:
        vals = results[key]
        avg = {k: np.mean([v[k] for v in vals]) for k in vals[0]}
        std = {k: np.std([v[k] for v in vals]) for k in vals[0]}
        print(f"{title:<26s} {avg['ACC']:>6.1f}±{std['ACC']:>3.1f}% {avg['PRE']:>6.1f}% {avg['SEN']:>6.1f}% {avg['SPE']:>6.1f}%")
    print("=" * 70)

    # 3. 绘制血流动力学响应对比图 (独立验证集)
    time_axis = np.arange(256) / 10.0 - 5.0
    real_val, synth_val, val_labels = real_all[val_idx], synth_all[val_idx], labels_np[val_idx]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, mask, title in [
        (axes[0], np.ones(len(val_labels), dtype=bool), f"Global Average (PCC: {val_cpcc:.4f})"),
        (axes[1], val_labels == 0, "Left Motor Imagery (LMI)"),
        (axes[2], val_labels == 1, "Right Motor Imagery (RMI)")
    ]:
        r_m = real_val[mask].mean(axis=(0, 1))
        s_m = synth_val[mask].mean(axis=(0, 1))
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.6, label='Onset (0s)')
        ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.4)
        ax.plot(time_axis, r_m, 'b-', linewidth=2.2, label='Real fNIRS')
        ax.plot(time_axis, s_m, 'r--', linewidth=2.2, label='Synthetic fNIRS')
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{modality} (Baseline-Corrected Z-score)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"SCDM Hemodynamic Response ({ckpt_name}) | Val MSE: {val_mse:.4f} | Curve PCC: {val_cpcc:.4f}", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(cfg["train"]["output_dir"], f"eval_curve_{epoch}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f">>> 血流动力学对比图已保存至: {plot_path}")

if __name__ == "__main__":
    evaluate_checkpoint(epoch="best", ddim_steps=50)