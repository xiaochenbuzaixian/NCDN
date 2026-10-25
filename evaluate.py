import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score
from data.shin_dataset import get_dataloaders
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

class DownstreamClassifier(nn.Module):
    """用于检验论文下游运动想象 (MI) 分类增益的标准多模态分类网络"""
    def __init__(self, mode="bimodal"):
        super().__init__()
        self.mode = mode
        if mode in ["eeg_only", "bimodal"]:
            self.eeg_net = nn.Sequential(
                nn.Conv1d(30, 32, kernel_size=15, stride=8, padding=7),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.AdaptiveAvgPool1d(16),
                nn.Flatten()
            )
        if mode in ["fnirs_only", "bimodal"]:
            self.fnirs_net = nn.Sequential(
                nn.Conv1d(36, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.AdaptiveAvgPool1d(16),
                nn.Flatten()
            )
        in_dim = 32 * 16 * 2 if mode == "bimodal" else 32 * 16
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, eeg, fnirs=None):
        if self.mode == "eeg_only":
            feat = self.eeg_net(eeg)
        elif self.mode == "fnirs_only":
            feat = self.fnirs_net(fnirs)
        else:
            feat = torch.cat([self.eeg_net(eeg), self.fnirs_net(fnirs)], dim=-1)
        return self.fc(feat)

def run_downstream_classification(train_e, train_f, train_y, val_e, val_f_real, val_f_syn, val_y, device):
    """训练分类器并对比：1.仅EEG  2.EEG+真实fNIRS  3.EEG+生成fNIRS"""
    results = {}
    for name, mode, tr_f, te_f in [
        ("仅 EEG (单模态基线)", "eeg_only", train_f, val_f_real),
        ("EEG + 真实 fNIRS (上限)", "bimodal", train_f, val_f_real),
        ("EEG + 合成 fNIRS (SCDM)", "bimodal", train_f, val_f_syn)
    ]:
        torch.manual_seed(42)
        clf = DownstreamClassifier(mode=mode).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-2)
        crit = nn.CrossEntropyLoss()

        # 快速训练分类器 30 轮
        clf.train()
        for _ in range(30):
            idx = torch.randperm(len(train_e))
            for i in range(0, len(train_e), 32):
                b = idx[i:i+32]
                opt.zero_grad()
                out = clf(train_e[b].to(device), tr_f[b].to(device))
                loss = crit(out, train_y[b].to(device))
                loss.backward()
                opt.step()

        clf.eval()
        with torch.no_grad():
            logits = clf(val_e.to(device), te_f.to(device))
            preds = logits.argmax(dim=-1).cpu().numpy()
            acc = accuracy_score(val_y.numpy(), preds) * 100
            f1 = f1_score(val_y.numpy(), preds, average="macro") * 100
            results[name] = (acc, f1)
    return results

def evaluate_checkpoint(epoch=100):
    with open("configs/shin_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    modality = cfg["data"]["target_modality"]
    processed_dir = cfg["data"]["processed_dir"]
    ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_ep{epoch}.pt")

    topo_dict = torch.load(os.path.join(processed_dir, f"topo_{modality.lower()}.pt"), map_location="cpu")
    topo_ef = topo_dict["topo_ef"].to(device)
    topo_fe = topo_dict["topo_fe"].to(device)

    train_loader, val_loader = get_dataloaders(
        processed_dir=processed_dir,
        modality=modality,
        batch_size=cfg["data"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"]
    )

    unet = SCDMUnet(time_dim=cfg["model"]["time_dim"]).to(device)
    model = SCDMDiffusion(unet, timesteps=cfg["model"]["timesteps"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print(f">>> 开始对 {ckpt_path} 进行全量验证集评估 (DDIM 50步快速采样)...")
    all_real, all_syn, val_e_list, val_y_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证集推理"):
            e_0 = batch["eeg"].to(device)
            f_0 = batch["fnirs"]
            syn_f = model.sample(e_0, topo_ef, topo_fe).cpu()

            all_real.append(f_0.numpy())
            all_syn.append(syn_f.numpy())
            val_e_list.append(batch["eeg"])
            val_y_list.append(batch["label"])

    all_real = np.concatenate(all_real, axis=0)
    all_syn = np.concatenate(all_syn, axis=0)

    # 1. 计算生成指标
    global_rmse = np.sqrt(np.mean((all_real - all_syn) ** 2))
    real_mean_curve = np.mean(all_real, axis=(0, 1))
    syn_mean_curve = np.mean(all_syn, axis=(0, 1))
    global_pcc, _ = pearsonr(real_mean_curve, syn_mean_curve)

    # 余弦相似度
    r_flat = real_mean_curve / (np.linalg.norm(real_mean_curve) + 1e-8)
    s_flat = syn_mean_curve / (np.linalg.norm(syn_mean_curve) + 1e-8)
    cos_sim = float(np.dot(r_flat, s_flat))

    print("\n" + "=" * 50)
    print(f"【1. 信号生成质量评估】 (Epoch {epoch} | 样本数: {len(all_real)})")
    print(f"  - 全量 RMSE       : {global_rmse:.4f}")
    print(f"  - 均值曲线 PCC    : {global_pcc:.4f}")
    print(f"  - 曲线余弦相似度  : {cos_sim:.4f}")

    # 2. 运行原论文同款下游分类测试
    train_e = torch.cat([b["eeg"] for b in train_loader], dim=0)
    train_f = torch.cat([b["fnirs"] for b in train_loader], dim=0)
    train_y = torch.cat([b["label"] for b in train_loader], dim=0)
    val_e = torch.cat(val_e_list, dim=0)
    val_y = torch.cat(val_y_list, dim=0)

    clf_res = run_downstream_classification(
        train_e, train_f, train_y,
        val_e, torch.tensor(all_real), torch.tensor(all_syn), val_y, device
    )
    print("\n【2. 下游运动想象 (MI) 分类任务检验】")
    for k, (acc, f1) in clf_res.items():
        print(f"  - {k:<24} | Acc: {acc:.2f}% | Macro-F1: {f1:.2f}%")
    print("=" * 50)

    # 3. 绘制对比曲线与误差带
    plt.figure(figsize=(10, 4.5))
    time_axis = np.linspace(-5, 20.6, 256)
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.8, label='Task Onset (0s)')
    plt.plot(time_axis, real_mean_curve, label='Real fNIRS (Ground Truth)', color='#1f77b4', linewidth=2.5)
    plt.plot(time_axis, syn_mean_curve, label='Synthetic fNIRS (SCDM)', color='#d62728', linestyle='--', linewidth=2.5)
    plt.title(f"Hemodynamic Response (Epoch {epoch}) | RMSE: {global_rmse:.4f} | PCC: {global_pcc:.4f}")
    plt.xlabel("Time relative to task onset (s)")
    plt.ylabel("Amplitude (Z-score)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(cfg["train"]["output_dir"], f"eval_curve_ep{epoch}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f">>> 对比曲线图已保存至: {plot_path}")

if __name__ == "__main__":
    evaluate_checkpoint(epoch=80)