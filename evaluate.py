import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from data.shin_dataset import get_dataloaders
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

def extract_eeg_bandpower_features(eeg_np, fs=160.0):
    """提取运动想象核心频段 (Mu: 8-13Hz, Beta: 13-30Hz) 的多时段对数功率特征"""
    N, C, L = eeg_np.shape
    nyq = 0.5 * fs
    sos_mu = butter(4, [8.0 / nyq, 13.0 / nyq], btype='bandpass', output='sos')
    sos_beta = butter(4, [13.0 / nyq, 30.0 / nyq], btype='bandpass', output='sos')

    # 取任务期 [0s ~ 15s] 对应采样点 [800:3200]
    task_eeg = eeg_np[:, :, 800:3200]
    mu_sig = sosfiltfilt(sos_mu, task_eeg, axis=-1)
    beta_sig = sosfiltfilt(sos_beta, task_eeg, axis=-1)

    # 分3个时间窗提取对数方差 (Log-Variance)
    feats = []
    for sig in [mu_sig, beta_sig]:
        for w in np.array_split(sig, 3, axis=-1):
            feats.append(np.log(np.var(w, axis=-1) + 1e-6))
    return np.concatenate(feats, axis=-1)  # [N, 30 * 6]

def extract_fnirs_hrf_features(fnirs_np):
    """
    参考 safii-74/src/data/hrf_features.py:
    提取任务期 [0s~20s] (采样点 50~256) 的 HRF 生理特征: 均值、峰值、斜率、标准差与窗口均值
    """
    task_f = fnirs_np[:, :, 50:]  # 排除前50点基线期
    f_mean = np.mean(task_f, axis=-1)
    f_max = np.max(task_f, axis=-1)
    f_min = np.min(task_f, axis=-1)
    # 血流上升期斜率 (50~150点) 与恢复期斜率 (150~256点)
    f_slope1 = (fnirs_np[:, :, 150] - fnirs_np[:, :, 50]) / 100.0
    f_slope2 = (fnirs_np[:, :, 255] - fnirs_np[:, :, 150]) / 105.0
    win_means = [np.mean(w, axis=-1) for w in np.array_split(task_f, 4, axis=-1)]
    return np.concatenate([f_mean, f_max, f_min, f_slope1, f_slope2] + win_means, axis=-1)

def run_downstream_classification(train_e, train_f, train_y, val_e, val_f_real, val_f_syn, val_y):
    """使用标准 BCI 特征工程 + SVM/RF 检验跨模态生成对下游 MI 分类的增益"""
    tr_eeg_feat = extract_eeg_bandpower_features(train_e)
    te_eeg_feat = extract_eeg_bandpower_features(val_e)

    tr_fnirs_feat = extract_fnirs_hrf_features(train_f)
    te_fnirs_real_feat = extract_fnirs_hrf_features(val_f_real)
    te_fnirs_syn_feat = extract_fnirs_hrf_features(val_f_syn)

    results = {}
    configs = [
        ("仅 EEG (单模态基线)", tr_eeg_feat, te_eeg_feat),
        ("EEG + 真实 fNIRS (上限)", np.hstack([tr_eeg_feat, tr_fnirs_feat]), np.hstack([te_eeg_feat, te_fnirs_real_feat])),
        ("EEG + 合成 fNIRS (SCDM)", np.hstack([tr_eeg_feat, tr_fnirs_feat]), np.hstack([te_eeg_feat, te_fnirs_syn_feat]))
    ]

    for name, X_tr, X_te in configs:
        scaler = StandardScaler()
        X_tr_s = np.nan_to_num(scaler.fit_transform(X_tr), 0.0)
        X_te_s = np.nan_to_num(scaler.transform(X_te), 0.0)

        clf = SVC(C=1.0, kernel='rbf', random_state=42)
        clf.fit(X_tr_s, train_y)
        preds = clf.predict(X_te_s)
        acc = accuracy_score(val_y, preds) * 100
        f1 = f1_score(val_y, preds, average="macro") * 100
        results[name] = (acc, f1)
    return results

def evaluate_checkpoint(epoch="best"):
    with open("configs/shin_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    modality = cfg["data"]["target_modality"]
    processed_dir = cfg["data"]["processed_dir"]

    # 修复 best 权重路径拼接 Bug
    if str(epoch) == "best":
        ckpt_name = f"scdm_{modality.lower()}_best.pt"
    else:
        ckpt_name = f"scdm_{modality.lower()}_ep{epoch}.pt"
    ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

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

    print(f">>> 开始对 {ckpt_path} 进行全量验证集评估...")
    all_real, all_syn, val_e_list, val_y_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证集推理"):
            e_0 = batch["eeg"].to(device)
            f_0 = batch["fnirs"]
            syn_f = model.sample(e_0, topo_ef, topo_fe).cpu()

            all_real.append(f_0.numpy())
            all_syn.append(syn_f.numpy())
            val_e_list.append(batch["eeg"].numpy())
            val_y_list.append(batch["label"].numpy())

    all_real = np.concatenate(all_real, axis=0)
    all_syn = np.concatenate(all_syn, axis=0)

    # 1. 计算生成指标
    global_rmse = np.sqrt(np.mean((all_real - all_syn) ** 2))
    real_mean_curve = np.mean(all_real, axis=(0, 1))
    syn_mean_curve = np.mean(all_syn, axis=(0, 1))
    global_pcc, _ = pearsonr(real_mean_curve, syn_mean_curve)

    r_flat = real_mean_curve / (np.linalg.norm(real_mean_curve) + 1e-8)
    s_flat = syn_mean_curve / (np.linalg.norm(syn_mean_curve) + 1e-8)
    cos_sim = float(np.dot(r_flat, s_flat))

    print("\n" + "=" * 55)
    print(f"【1. 信号生成质量评估】 (权重: {ckpt_name} | 样本数: {len(all_real)})")
    print(f"  - 全量 RMSE       : {global_rmse:.4f}")
    print(f"  - 均值曲线 PCC    : {global_pcc:.4f}")
    print(f"  - 曲线余弦相似度  : {cos_sim:.4f}")

    # 2. 下游分类评估
    train_e = np.concatenate([b["eeg"].numpy() for b in train_loader], axis=0)
    train_f = np.concatenate([b["fnirs"].numpy() for b in train_loader], axis=0)
    train_y = np.concatenate([b["label"].numpy() for b in train_loader], axis=0)
    val_e = np.concatenate(val_e_list, axis=0)
    val_y = np.concatenate(val_y_list, axis=0)

    clf_res = run_downstream_classification(train_e, train_f, train_y, val_e, all_real, all_syn, val_y)
    print("\n【2. 下游运动想象 (MI) 分类任务检验 (CSP/Bandpower + HRF Features)】")
    for k, (acc, f1) in clf_res.items():
        print(f"  - {k:<24} | Acc: {acc:.2f}% | Macro-F1: {f1:.2f}%")
    print("=" * 55)

    # 3. 绘制血流动力学对比曲线
    plt.figure(figsize=(10, 4.5))
    time_axis = np.linspace(-5, 20.6, 256)
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.8, label='Task Onset (0s)')
    plt.axhline(y=0, color='black', linewidth=0.8, alpha=0.4)
    plt.plot(time_axis, real_mean_curve, label='Real fNIRS (Ground Truth)', color='#1f77b4', linewidth=2.5)
    plt.plot(time_axis, syn_mean_curve, label='Synthetic fNIRS (SCDM)', color='#d62728', linestyle='--', linewidth=2.5)
    plt.title(f"Hemodynamic Response ({ckpt_name}) | RMSE: {global_rmse:.4f} | PCC: {global_pcc:.4f}")
    plt.xlabel("Time relative to task onset (s)")
    plt.ylabel("Baseline-Corrected Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(cfg["train"]["output_dir"], f"eval_curve_{epoch}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f">>> 对比曲线图已保存至: {plot_path}")

if __name__ == "__main__":
    evaluate_checkpoint(epoch="best")