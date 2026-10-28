import os
import json
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, ttest_rel, t as student_t
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

# ==========================================
# 1. 保留完整 Mu/Beta (8-30Hz) 频带的 80Hz EEGNet
# ==========================================
class EEGNet(nn.Module):
    """
    80Hz EEGNet (downsample=2, Nyquist=40Hz):
    完整保留运动想象核心的 Mu (8-13Hz) 与 Beta (13-30Hz) 频段信息
    """
    def __init__(self, downsample=2):
        super().__init__()
        self.downsample = downsample
        n_in = 4000 // downsample  # 2000 pts @ 80Hz
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 16, (30, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        self._feat_dim = 16 * (n_in // 32)
        self.fc = nn.Linear(self._feat_dim, 2)

    def extract_features(self, x):
        if self.downsample > 1:
            x = F.avg_pool1d(x, kernel_size=self.downsample, stride=self.downsample)
        x = self.bn1(self.conv1(x.unsqueeze(1)))
        x = self.drop1(self.pool1(F.elu(self.bn2(self.depthwise(x)))))
        return self.separable(x).flatten(1)

    def forward(self, x):
        return self.fc(self.extract_features(x))

# ==========================================
# 2. 结合头皮物理坐标的半球偏侧化 fNIRS 分支与防 BN 污染的 HybridNet
# ==========================================
def get_motor_hemisphere_indices(raw_dir="datasets/raw"):
    nirs_mnt_path = os.path.join(raw_dir, "fNIRS", "subject 01", "mnt.mat")
    if os.path.exists(nirs_mnt_path):
        mnt = sio.loadmat(nirs_mnt_path, struct_as_record=False, squeeze_me=True)['mnt']
        nx, ny = np.array(mnt.x).flatten()[:36], np.array(mnt.y).flatten()[:36]
        left_idx = np.where((nx < -0.03) & (ny < 0.45))[0]
        right_idx = np.where((nx > 0.03) & (ny < 0.45))[0]
        if len(left_idx) > 0 and len(right_idx) > 0:
            return left_idx, right_idx
    return np.arange(0, 12), np.arange(12, 24)

class HemisphericfNIRSBranch(nn.Module):
    """提取 fNIRS 多时间窗响应、上升斜率以及左/右运动皮层偏侧化不对称特征"""
    def __init__(self, left_idx, right_idx, n_channels=36, out_dim=32):
        super().__init__()
        self.register_buffer("left_idx", torch.tensor(left_idx, dtype=torch.long))
        self.register_buffer("right_idx", torch.tensor(right_idx, dtype=torch.long))

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=11, padding=5, stride=2, groups=n_channels),
            nn.Conv1d(n_channels, 24, kernel_size=1),
            nn.BatchNorm1d(24),
            nn.ELU(),
            nn.Dropout(0.35),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten()
        )
        # 统计量: 3阶段均值(36*3) + 2斜率(36*2) + 半球对称差值(6) = 186
        in_dim = 24 * 4 + 36 * 5 + 6
        self.proj = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(0.4),
            nn.Linear(in_dim, out_dim),
            nn.ELU()
        )

    def forward(self, f):
        base = f[:, :, :50].mean(dim=-1, keepdim=True)
        df = f - base

        w1 = df[:, :, 50:100].mean(dim=-1)   # 0~5s
        w2 = df[:, :, 100:175].mean(dim=-1)  # 5~12.5s
        w3 = df[:, :, 175:245].mean(dim=-1)  # 12.5~19.5s
        s_up = w2 - w1
        s_dn = w3 - w2

        # 左/右半球运动皮层偏侧化对比 (Contralateral vs. Ipsilateral Asymmetry)
        L1, R1 = w1[:, self.left_idx].mean(dim=-1, keepdim=True), w1[:, self.right_idx].mean(dim=-1, keepdim=True)
        L2, R2 = w2[:, self.left_idx].mean(dim=-1, keepdim=True), w2[:, self.right_idx].mean(dim=-1, keepdim=True)
        L3, R3 = w3[:, self.left_idx].mean(dim=-1, keepdim=True), w3[:, self.right_idx].mean(dim=-1, keepdim=True)
        Ls, Rs = s_up[:, self.left_idx].mean(dim=-1, keepdim=True), s_up[:, self.right_idx].mean(dim=-1, keepdim=True)
        lat_feats = torch.cat([L1 - R1, L2 - R2, L3 - R3, Ls - Rs, L2, R2], dim=-1)

        c_feat = self.conv(df[:, :, 50:])
        return self.proj(torch.cat([c_feat, w1, w2, w3, s_up, s_dn, lat_feats], dim=-1))

class HybridNet(nn.Module):
    """
    严格固定 EEG 评估态的双模态融合网络:
    重写 train() 方法，确保冻结的 self.eeg 永远处于 eval() 模式，绝不污染 BatchNorm 运行统计量
    支持两种融合模式消融: 'concat_head' (特征拼接融合头) 与 'logit_sum' (标准化 Logit 加权融合)
    """
    def __init__(self, pretrained_eegnet: EEGNet, left_idx, right_idx, fusion_mode="concat_head"):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.eeg = copy.deepcopy(pretrained_eegnet)
        self.eeg.eval()
        for p in self.eeg.parameters():
            p.requires_grad = False

        self.fnirs_branch = HemisphericfNIRSBranch(left_idx, right_idx, n_channels=36, out_dim=32)
        self.fnirs_fc = nn.Linear(32, 2)
        self.concat_fc = nn.Sequential(
            nn.BatchNorm1d(self.eeg._feat_dim + 32),
            nn.Dropout(0.35),
            nn.Linear(self.eeg._feat_dim + 32, 32),
            nn.ELU(),
            nn.Linear(32, 2)
        )
        self.e_norm = nn.BatchNorm1d(2, affine=False)
        self.f_norm = nn.BatchNorm1d(2, affine=False)

    def train(self, mode: bool = True):
        super().train(mode)
        # 关键修复：无论外部如何调用 model.train()，冻结的 EEG 分支永远保持 eval() 状态！
        self.eeg.eval()
        return self

    def forward(self, e, f):
        with torch.no_grad():
            e_feat = self.eeg.extract_features(e)
            e_logits = self.eeg.fc(e_feat)

        f_feat = self.fnirs_branch(f)
        f_logits = self.fnirs_fc(f_feat)
        bi_logits = self.concat_fc(torch.cat([e_feat, f_feat], dim=1))

        # 通过无参 BatchNorm1d 将 EEG Logits 与 fNIRS Logits 对齐到相同方差尺度，打破 EEG 训练集置信度垄断
        e_z = self.e_norm(e_logits)
        f_z = self.f_norm(0.5 * f_logits + 0.5 * bi_logits)

        if self.fusion_mode == "logit_sum":
            total_logits = 0.65 * e_z + 0.35 * f_z
        else:
            total_logits = 0.60 * e_z + 0.40 * f_z
        return total_logits, f_logits, bi_logits

def classification_metrics(y_true, y_pred):
    return {
        "ACC": accuracy_score(y_true, y_pred) * 100,
        "PRE": precision_score(y_true, y_pred, pos_label=0, zero_division=0) * 100,
        "SEN": recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100,
        "SPE": recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
    }

def train_eegnet(tr_loader, te_loader, epochs=50, lr=1e-3, device="cuda", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = EEGNet(downsample=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for e, y in tr_loader:
            loss = crit(model(e.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for e, y in te_loader:
            preds.append(model(e.to(device)).argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    return model, classification_metrics(np.concatenate(targets), np.concatenate(preds))

def train_hybrid(base_model, tr_loader, te_loader, left_idx, right_idx, fusion_mode="concat_head", epochs=40, lr=1e-3, device="cuda", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HybridNet(base_model, left_idx, right_idx, fusion_mode=fusion_mode).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()

    model.train()  # 内部已重写，self.eeg 严格保持 eval()
    for _ in range(epochs):
        for e, f, y in tr_loader:
            e, f, y = e.to(device), f.to(device), y.to(device)
            total_logits, f_logits, bi_logits = model(e, f)
            # 独立督促 f_logits 与 bi_logits 学习血氧类别信息
            loss = crit(total_logits, y) + 0.5 * crit(f_logits, y) + 0.5 * crit(bi_logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for e, f, y in te_loader:
            total_logits, _, _ = model(e.to(device), f.to(device))
            preds.append(total_logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    return classification_metrics(np.concatenate(targets), np.concatenate(preds))

def calc_raw_generation_metrics(real_arr, synth_arr):
    """不借用任何测试集统计量做缩放，100% 原始输出直接评估 (零泄漏)"""
    mse = float(np.mean((real_arr - synth_arr) ** 2))
    rmse = float(np.sqrt(mse))
    corrs = [
        float(((r - r.mean()) * (s - s.mean())).sum() /
              (np.sqrt(((r - r.mean()) ** 2).sum() * ((s - s.mean()) ** 2).sum()) + 1e-12))
        for r, s in zip(real_arr, synth_arr)
    ]
    curve_pcc, _ = pearsonr(real_arr.mean(axis=(0, 1)), synth_arr.mean(axis=(0, 1)))
    return mse, rmse, float(np.mean(corrs)), float(curve_pcc)

def analyze_physiological_fidelity(real_test, synth_test, labels_test, left_idx, right_idx):
    """检查试次级生理特征分布: 逐通道方差比、峰值、峰值潜伏期、以及 LMI vs RMI 半球偏侧化差异"""
    r_task, s_task = real_test[:, :, 50:], synth_test[:, :, 50:]
    r_var, s_var = float(np.var(r_task)), float(np.var(s_task))
    r_peak = float(np.max(np.abs(r_task), axis=-1).mean())
    s_peak = float(np.max(np.abs(s_task), axis=-1).mean())
    r_lat = float((np.argmax(np.abs(r_task), axis=-1).mean() / 10.0))
    s_lat = float((np.argmax(np.abs(s_task), axis=-1).mean() / 10.0))

    # 计算 LMI (0) 与 RMI (1) 下的左右半球差值 (Left - Right)
    lmi_m, rmi_m = (labels_test == 0), (labels_test == 1)
    r_lat_lmi = float((r_task[lmi_m][:, left_idx].mean() - r_task[lmi_m][:, right_idx].mean()))
    r_lat_rmi = float((r_task[rmi_m][:, left_idx].mean() - r_task[rmi_m][:, right_idx].mean()))
    s_lat_lmi = float((s_task[lmi_m][:, left_idx].mean() - s_task[lmi_m][:, right_idx].mean()))
    s_lat_rmi = float((s_task[rmi_m][:, left_idx].mean() - s_task[rmi_m][:, right_idx].mean()))

    return {
        "var_ratio": f"Real={r_var:.4f}, Synth={s_var:.4f}",
        "peak_amp": f"Real={r_peak:.4f}, Synth={s_peak:.4f}",
        "peak_time_s": f"Real={r_lat:.2f}s, Synth={s_lat:.2f}s",
        "hemi_contrast_real": r_lat_lmi - r_lat_rmi,
        "hemi_contrast_synth": s_lat_lmi - s_lat_rmi
    }

def paired_stat_summary(acc_base, acc_comp):
    diffs = np.array(acc_comp) - np.array(acc_base)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    n = len(diffs)
    se = std_d / np.sqrt(n)
    ci95 = float(student_t.ppf(0.975, df=n - 1) * se)
    _, p_val = ttest_rel(acc_comp, acc_base)
    return mean_d, ci95, float(p_val), diffs

def evaluate_checkpoint(epoch="best", ddim_steps=50):
    with open("configs/shin_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    modality = cfg["data"]["target_modality"]
    processed_dir = cfg["data"]["processed_dir"]
    ckpt_name = f"scdm_{modality.lower()}_best.pt" if str(epoch) == "best" else f"scdm_{modality.lower()}_ep{epoch}.pt"

    topo_dict = torch.load(os.path.join(processed_dir, f"topo_{modality.lower()}.pt"), map_location="cpu")
    topo_ef, topo_fe = topo_dict["topo_ef"].to(device), topo_dict["topo_fe"].to(device)
    corr_ef = topo_dict.get("corr_ef", None)
    if corr_ef is not None:
        corr_ef = corr_ef.to(device)

    unet = SCDMUnet(
        time_dim=cfg["model"].get("time_dim", 64),
        base_channels=cfg["model"].get("base_channels", 32),
        n_blocks=6
    ).to(device)
    model = SCDMDiffusion(unet, timesteps=cfg["model"]["timesteps"]).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg["train"]["checkpoint_dir"], ckpt_name), map_location=device))
    model.eval()

    eeg_cond = torch.load(os.path.join(processed_dir, "eeg_hrf_epochs.pt"))
    eeg_np = torch.load(os.path.join(processed_dir, "eeg_epochs.pt")).numpy()
    real_target = torch.load(os.path.join(processed_dir, f"fnirs_{modality.lower()}_epochs.pt")).numpy()
    labels_np = torch.load(os.path.join(processed_dir, "labels.pt")).numpy()

    with open(os.path.join(processed_dir, "split_indices.json"), "r") as f:
        split_info = json.load(f)
    val_idx = np.array(split_info["val_indices"], dtype=int)
    test_idx = np.array(split_info["test_indices"], dtype=int)
    cv_folds = split_info["cv_5fold"]

    print(f">>> 正在使用 6-Block 标准 SCDM DDIM-{ddim_steps} 步采样生成合成 fNIRS (无测试集幅值校准)...")
    synth_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(eeg_cond), 16), desc="DDIM 采样"):
            synth_list.append(
                model.sample(eeg_cond[i:i + 16].to(device), topo_ef, topo_fe, steps=ddim_steps, corr_ef=corr_ef).cpu().numpy()
            )
    synth_all = np.concatenate(synth_list, axis=0)

    v_mse, v_rmse, v_spcc, v_cpcc = calc_raw_generation_metrics(real_target[val_idx], synth_all[val_idx])
    t_mse, t_rmse, t_spcc, t_cpcc = calc_raw_generation_metrics(real_target[test_idx], synth_all[test_idx])

    left_idx, right_idx = get_motor_hemisphere_indices("datasets/raw")
    physio_stats = analyze_physiological_fidelity(real_target[test_idx], synth_all[test_idx], labels_np[test_idx], left_idx, right_idx)

    print("\n" + "=" * 80)
    print(f"【1. 信号生成质量与试次级生理特征评估 (零测试集泄漏)】 (权重: {ckpt_name})")
    print(f"  [独立验证集 N={len(val_idx)}] MSE: {v_mse:.4f} (RMSE: {v_rmse:.4f}) | 单试次 PCC: {v_spcc:.4f} | 曲线 PCC: {v_cpcc:.4f}")
    print(f"  [留出测试集 N={len(test_idx)}] MSE: {t_mse:.4f} (RMSE: {t_rmse:.4f}) | 单试次 PCC: {t_spcc:.4f} | 曲线 PCC: {t_cpcc:.4f}")
    print(f"  [生理分布检验] 任务期方差: {physio_stats['var_ratio']} | 均值峰值: {physio_stats['peak_amp']} | 达峰时间: {physio_stats['peak_time_s']}")
    print(f"  [类别偏侧化度] 左/右运动区 LMI-RMI 不对称对比度: Real={physio_stats['hemi_contrast_real']:+.4f}, Synth={physio_stats['hemi_contrast_synth']:+.4f}")

    print("\n>>> 正在运行严格 5 折交叉验证 (Stratified 5-Fold CV) 与配对统计检验...")
    results = {"eeg_only": [], "eeg_real": [], "eeg_synth": []}

    for fold_i, fold in enumerate(cv_folds):
        tr, te = np.array(fold["train"], dtype=int), np.array(fold["test"], dtype=int)
        e_tr, e_te = torch.from_numpy(eeg_np[tr]).float(), torch.from_numpy(eeg_np[te]).float()
        f_tr, f_te = torch.from_numpy(real_target[tr]).float(), torch.from_numpy(real_target[te]).float()
        s_tr, s_te = torch.from_numpy(synth_all[tr]).float(), torch.from_numpy(synth_all[te]).float()
        y_tr, y_te = torch.from_numpy(labels_np[tr]), torch.from_numpy(labels_np[te])

        base_model, m_eeg = train_eegnet(
            DataLoader(TensorDataset(e_tr, y_tr), 32, shuffle=True),
            DataLoader(TensorDataset(e_te, y_te), 64),
            seed=fold_i * 100 + 42
        )
        m_real = train_hybrid(
            base_model,
            DataLoader(TensorDataset(e_tr, f_tr, y_tr), 32, shuffle=True),
            DataLoader(TensorDataset(e_te, f_te, y_te), 64),
            left_idx, right_idx, fusion_mode="concat_head", seed=fold_i * 100 + 42
        )
        m_syn = train_hybrid(
            base_model,
            DataLoader(TensorDataset(e_tr, s_tr, y_tr), 32, shuffle=True),
            DataLoader(TensorDataset(e_te, s_te, y_te), 64),
            left_idx, right_idx, fusion_mode="concat_head", seed=fold_i * 100 + 42
        )
        results["eeg_only"].append(m_eeg)
        results["eeg_real"].append(m_real)
        results["eeg_synth"].append(m_syn)
        print(f"  Fold {fold_i+1}/5 | EEG-only: {m_eeg['ACC']:.2f}% | EEG+Real: {m_real['ACC']:.2f}% ({m_real['ACC']-m_eeg['ACC']:+.2f}%) | EEG+Synth: {m_syn['ACC']:.2f}% ({m_syn['ACC']-m_eeg['ACC']:+.2f}%)")

    print("\n【2. 下游运动想象 (MI) 严格 5 折交叉验证结果与配对显著性检验】")
    print(f"{'Mode':<26s} {'ACC':>12s} {'PRE':>10s} {'SEN':>10s} {'SPE':>10s}")
    print("-" * 80)
    for k, title in [
        ("eeg_only", "EEG-only (80Hz EEGNet)"),
        ("eeg_real", "EEG + Real fNIRS"),
        ("eeg_synth", "EEG + Synth fNIRS (SCDM)")
    ]:
        avg = {m: np.mean([v[m] for v in results[k]]) for m in results[k][0]}
        std = {m: np.std([v[m] for v in results[k]], ddof=1) for m in results[k][0]}
        print(f"{title:<26s} {avg['ACC']:>6.2f}±{std['ACC']:>4.2f}% {avg['PRE']:>9.2f}% {avg['SEN']:>9.2f}% {avg['SPE']:>9.2f}%")

    acc_e = [v["ACC"] for v in results["eeg_only"]]
    acc_r = [v["ACC"] for v in results["eeg_real"]]
    acc_s = [v["ACC"] for v in results["eeg_synth"]]
    r_diff, r_ci, r_p, _ = paired_stat_summary(acc_e, acc_r)
    s_diff, s_ci, s_p, _ = paired_stat_summary(acc_e, acc_s)

    print("-" * 80)
    print(f"  ★ 配对增益 (EEG+Real  vs EEG-only): ΔACC = {r_diff:+.2f}% ± {r_ci:.2f}% (95% CI) | paired t-test p = {r_p:.4f}")
    print(f"  ★ 配对增益 (EEG+Synth vs EEG-only): ΔACC = {s_diff:+.2f}% ± {s_ci:.2f}% (95% CI) | paired t-test p = {s_p:.4f}")
    print("=" * 80)

    # 绘制留出测试集上的血流动力学响应对比图
    time_axis = np.arange(256) / 10.0 - 5.0
    r_te, s_te, y_te_arr = real_target[test_idx], synth_all[test_idx], labels_np[test_idx]
    plt.figure(figsize=(16, 4.5))
    for i, (m, t) in enumerate([
        (np.ones(len(test_idx), dtype=bool), f"Held-out Test Global (PCC: {t_cpcc:.4f})"),
        (y_te_arr == 0, "Left Motor Imagery (LMI)"),
        (y_te_arr == 1, "Right Motor Imagery (RMI)")
    ]):
        ax = plt.subplot(1, 3, i + 1)
        ax.plot(time_axis, r_te[m].mean(axis=(0, 1)), 'b-', lw=2.2, label='Real fNIRS')
        ax.plot(time_axis, s_te[m].mean(axis=(0, 1)), 'r--', lw=2.2, label='Synth fNIRS')
        ax.axvline(0, color='k', ls=':', alpha=0.5, label='Onset (0s)')
        ax.axhline(0, color='gray', lw=0.8, alpha=0.4)
        ax.set_title(t); ax.set_xlabel("Time (s)"); ax.set_ylabel(f"{modality} (Z-score)")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["train"]["output_dir"], f"eval_curve_best.png"), dpi=300, bbox_inches='tight')
    print(f">>> 留出测试集血流动力学对比图已保存至: {os.path.join(cfg['train']['output_dir'], 'eval_curve_best.png')}")

if __name__ == "__main__":
    evaluate_checkpoint(epoch="best", ddim_steps=50)