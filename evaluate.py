import os
import csv
import json
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, ttest_rel
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

class EEGNet(nn.Module):
    """80Hz EEGNet (downsample=2, Nyquist=40Hz): 完整保留 Mu (8-13Hz) 与 Beta (13-30Hz)"""
    def __init__(self, downsample=2):
        super().__init__()
        self.downsample = downsample
        n_in = 4000 // downsample
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

def ols_slope(seg):
    L = seg.shape[-1]
    t = np.linspace(-1.0, 1.0, L, dtype=np.float32)
    return np.tensordot(seg, t, axes=([-1], [0])) / np.sum(t ** 2)

def extract_fnirs_motor_features(f_sig, sub_ids, tr_idx, left_idx, right_idx):
    """
    提取单模态 fNIRS 运动皮层特征 (仅使用训练集 tr_idx 统计量消除被试固有解剖偏置，100% 零测试泄漏):
    1. 减去任务前 [-5s, 0s] 基线
    2. 减去该被试在训练集 (tr_idx) 上的平均血氧响应模板 (消除跨被试解剖学非对称偏移，凸显单试次 LMI vs RMI 偏侧化!)
    """
    df = f_sig - f_sig[:, :, :50].mean(axis=-1, keepdims=True)
    df_centered = np.copy(df)
    for s_id in np.unique(sub_ids):
        s_mask = (sub_ids == s_id)
        s_tr = np.intersect1d(np.where(s_mask)[0], tr_idx)
        if len(s_tr) > 0:
            sub_template = df[s_tr].mean(axis=0, keepdims=True)
            df_centered[s_mask] = df[s_mask] - sub_template

    n_pairs = min(len(left_idx), len(right_idx))
    l_sub, r_sub = left_idx[:n_pairs], right_idx[:n_pairs]

    # 提取关键生理窗口: [1-5s], [4-9s], [7-13s], [11-17s], [15-20s]
    w1 = df_centered[:, :, 60:100].mean(axis=-1)
    w2 = df_centered[:, :, 90:140].mean(axis=-1)
    w3 = df_centered[:, :, 120:180].mean(axis=-1)
    w4 = df_centered[:, :, 160:220].mean(axis=-1)
    w5 = df_centered[:, :, 200:250].mean(axis=-1)

    # 最小二乘斜率与峰值面积
    s1 = ols_slope(df_centered[:, :, 55:135])
    s2 = ols_slope(df_centered[:, :, 100:185])
    s3 = ols_slope(df_centered[:, :, 160:240])

    # 左右运动皮层对称光极对差值
    lat_w2 = w2[:, l_sub] - w2[:, r_sub]
    lat_w3 = w3[:, l_sub] - w3[:, r_sub]
    lat_w4 = w4[:, l_sub] - w4[:, r_sub]
    lat_s1 = s1[:, l_sub] - s1[:, r_sub]
    lat_s2 = s2[:, l_sub] - s2[:, r_sub]

    hemi_summary = np.column_stack([
        w2[:, left_idx].mean(axis=-1) - w2[:, right_idx].mean(axis=-1),
        w3[:, left_idx].mean(axis=-1) - w3[:, right_idx].mean(axis=-1),
        w4[:, left_idx].mean(axis=-1) - w4[:, right_idx].mean(axis=-1),
        s1[:, left_idx].mean(axis=-1) - s1[:, right_idx].mean(axis=-1),
    ])

    return np.concatenate([w1, w2, w3, w4, w5, s1, s2, s3, lat_w2, lat_w3, lat_w4, lat_s1, lat_s2, hemi_summary], axis=-1), df_centered

class TemporalSpatialfNIRSNet(nn.Module):
    """纯单模态 fNIRS 深度特征提取器"""
    def __init__(self, stat_dim, in_ch=36, out_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=11, padding=5, stride=2, groups=in_ch),
            nn.Conv1d(in_ch, 24, kernel_size=1),
            nn.BatchNorm1d(24),
            nn.ELU(),
            nn.Dropout(0.35),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten()
        )
        self.proj = nn.Sequential(
            nn.BatchNorm1d(24 * 4 + stat_dim),
            nn.Dropout(0.4),
            nn.Linear(24 * 4 + stat_dim, out_dim),
            nn.ELU()
        )
        self.fc = nn.Linear(out_dim, 2)

    def extract_feat(self, f_c, f_stat):
        c_feat = self.conv(f_c[:, :, 50:])
        return self.proj(torch.cat([c_feat, f_stat], dim=-1))

    def forward(self, f_c, f_stat):
        return self.fc(self.extract_feat(f_c, f_stat))

class GatedHybridNet(nn.Module):
    """
    严格固定 EEG 评估态的跨模态门控网络:
    重写 train() 方法确保 self.eeg 永远保持 eval()，利用跨模态注意力门控融合特征
    """
    def __init__(self, pretrained_eegnet: EEGNet, stat_dim: int):
        super().__init__()
        self.eeg = copy.deepcopy(pretrained_eegnet)
        self.eeg.eval()
        for p in self.eeg.parameters():
            p.requires_grad = False

        self.fnirs_net = TemporalSpatialfNIRSNet(stat_dim=stat_dim, in_ch=36, out_dim=32)
        self.cross_gate = nn.Sequential(
            nn.Linear(self.eeg._feat_dim + 32, 32),
            nn.Sigmoid()
        )
        self.fusion_fc = nn.Sequential(
            nn.BatchNorm1d(self.eeg._feat_dim + 32),
            nn.Dropout(0.35),
            nn.Linear(self.eeg._feat_dim + 32, 32),
            nn.ELU(),
            nn.Linear(32, 2)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.eeg.eval()
        return self

    def forward(self, e, f_c, f_stat):
        with torch.no_grad():
            e_feat = self.eeg.extract_features(e)
            e_logits = self.eeg.fc(e_feat)
        f_feat = self.fnirs_net.extract_feat(f_c, f_stat)
        f_logits = self.fnirs_net.fc(f_feat)

        gate = self.cross_gate(torch.cat([e_feat, f_feat], dim=1))
        bi_logits = self.fusion_fc(torch.cat([e_feat, f_feat * gate], dim=1))
        return e_logits, f_logits, bi_logits

def classification_metrics(y_true, y_pred):
    return {
        "ACC": accuracy_score(y_true, y_pred) * 100,
        "PRE": precision_score(y_true, y_pred, pos_label=0, zero_division=0) * 100,
        "SEN": recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100,
        "SPE": recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
    }

def train_eeg_baseline(e_tr, y_tr, e_val, y_val, e_te, device="cuda", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = EEGNet(downsample=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
    crit = nn.CrossEntropyLoss()

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(e_tr).float(), torch.from_numpy(y_tr)), batch_size=32, shuffle=True)
    best_val_acc, best_state = -1.0, None

    for _ in range(50):
        model.train()
        for eb, yb in tr_loader:
            loss = crit(model(eb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_lg = model(torch.from_numpy(e_val).float().to(device)).cpu().numpy()
            val_acc = accuracy_score(y_val, val_lg.argmax(axis=1))
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr_lg = model(torch.from_numpy(e_tr).float().to(device)).cpu().numpy()
        val_lg = model(torch.from_numpy(e_val).float().to(device)).cpu().numpy()
        te_lg = model(torch.from_numpy(e_te).float().to(device)).cpu().numpy()
    return model, tr_lg, val_lg, te_lg

def train_modality_and_gated_fusion(base_eeg, e_tr_lg, e_val_lg, e_te_lg,
                                    e_tr, e_val, e_te,
                                    f_tr, f_val, f_te,
                                    s_tr, s_val, s_te,
                                    y_tr, y_val, device="cuda", seed=42):
    """
    100% 对称地训练 fNIRS-only 与 GatedHybridNet:
    在验证集上搜索最优边界不确定性门控参数 (tau*, alpha*):
    - 当 EEG 置信度极高 (|e_z| > tau*) 时保持 EEG 决策不被低信噪比慢波干扰;
    - 当 EEG 处于模糊边界 (|e_z| <= tau*) 时引入 fNIRS 与跨模态融合分支协同修正!
    """
    torch.manual_seed(seed); np.random.seed(seed)
    scaler = StandardScaler()
    s_tr_n = np.nan_to_num(scaler.fit_transform(s_tr), 0.0)
    s_val_n = np.nan_to_num(scaler.transform(s_val), 0.0)
    s_te_n = np.nan_to_num(scaler.transform(s_te), 0.0)

    # 1. 纯 fNIRS 线性判别专家 (Shrinkage LDA + Ridge + Logistic)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(s_tr_n, y_tr)
    lda_tr, lda_val, lda_te = lda.decision_function(s_tr_n), lda.decision_function(s_val_n), lda.decision_function(s_te_n)

    ridge = RidgeClassifierCV(alphas=np.logspace(0, 4, 25))
    ridge.fit(s_tr_n, y_tr)
    rg_tr, rg_val, rg_te = ridge.decision_function(s_tr_n), ridge.decision_function(s_val_n), ridge.decision_function(s_te_n)

    lr_clf = LogisticRegression(C=0.02, max_iter=500, random_state=seed)
    lr_clf.fit(s_tr_n, y_tr)
    lr_tr, lr_val, lr_te = lr_clf.decision_function(s_tr_n), lr_clf.decision_function(s_val_n), lr_clf.decision_function(s_te_n)

    # 2. 训练 GatedHybridNet
    hyb = GatedHybridNet(base_eeg, stat_dim=s_tr_n.shape[1]).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, hyb.parameters()), lr=8e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
    crit = nn.CrossEntropyLoss()

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(e_tr).float(), torch.from_numpy(f_tr).float(), torch.from_numpy(s_tr_n).float(), torch.from_numpy(y_tr)), batch_size=32, shuffle=True)
    best_vh, best_sth = -1.0, None
    for _ in range(40):
        hyb.train()
        for eb, fb, sb, yb in dl_tr:
            _, f_lg, bi_lg = hyb(eb.to(device), fb.to(device), sb.to(device))
            loss = crit(bi_lg, yb.to(device)) + 0.5 * crit(f_lg, yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        hyb.eval()
        with torch.no_grad():
            _, f_v, bi_v = hyb(torch.from_numpy(e_val).float().to(device), torch.from_numpy(f_val).float().to(device), torch.from_numpy(s_val_n).float().to(device))
            va = accuracy_score(y_val, (0.5 * f_v + 0.5 * bi_v).argmax(dim=1).cpu().numpy())
            if va >= best_vh:
                best_vh = va; best_sth = copy.deepcopy(hyb.state_dict())

    hyb.load_state_dict(best_sth); hyb.eval()
    with torch.no_grad():
        _, f_tr_t, bi_tr_t = hyb(torch.from_numpy(e_tr).float().to(device), torch.from_numpy(f_tr).float().to(device), torch.from_numpy(s_tr_n).float().to(device))
        _, f_val_t, bi_val_t = hyb(torch.from_numpy(e_val).float().to(device), torch.from_numpy(f_val).float().to(device), torch.from_numpy(s_val_n).float().to(device))
        _, f_te_t, bi_te_t = hyb(torch.from_numpy(e_te).float().to(device), torch.from_numpy(f_te).float().to(device), torch.from_numpy(s_te_n).float().to(device))

    def z_norm(v, ref):
        return (v - np.mean(ref)) / (np.std(ref) + 1e-8)

    f_d_tr = np.diff(f_tr_t.cpu().numpy(), axis=1).ravel()
    f_d_val = np.diff(f_val_t.cpu().numpy(), axis=1).ravel()
    f_d_te = np.diff(f_te_t.cpu().numpy(), axis=1).ravel()

    bi_d_tr = np.diff(bi_tr_t.cpu().numpy(), axis=1).ravel()
    bi_d_val = np.diff(bi_val_t.cpu().numpy(), axis=1).ravel()
    bi_d_te = np.diff(bi_te_t.cpu().numpy(), axis=1).ravel()

    e_d_tr = e_tr_lg[:, 1] - e_tr_lg[:, 0]
    e_z_val = z_norm(e_val_lg[:, 1] - e_val_lg[:, 0], e_d_tr)
    e_z_te  = z_norm(e_te_lg[:, 1] - e_te_lg[:, 0], e_d_tr)

    # 纯 fNIRS 综合判别分数
    f_only_val = 0.35 * z_norm(lda_val, lda_tr) + 0.35 * z_norm(rg_val, rg_tr) + 0.15 * z_norm(lr_val, lr_tr) + 0.15 * z_norm(f_d_val, f_d_tr)
    f_only_te  = 0.35 * z_norm(lda_te, lda_tr)  + 0.35 * z_norm(rg_te, rg_tr)  + 0.15 * z_norm(lr_te, lr_tr)  + 0.15 * z_norm(f_d_te, f_d_tr)

    # 跨模态协同候选分数
    synergy_val = 0.55 * f_only_val + 0.45 * z_norm(bi_d_val, bi_d_tr)
    synergy_te  = 0.55 * f_only_te  + 0.45 * z_norm(bi_d_te, bi_d_tr)

    # 在验证集上网格搜索最优边界门控参数 (tau: 边界宽度, alpha: 辅助权重)
    base_val_acc = accuracy_score(y_val, (e_z_val > 0.0).astype(np.int64))
    best_val_acc = base_val_acc
    best_alpha, best_tau = 0.0, 0.5

    for tau in [0.25, 0.40, 0.60, 0.85, 1.20, 2.00]:
        gate_val = np.exp(-0.5 * (e_z_val / tau) ** 2)
        for alpha in np.linspace(0.0, 0.60, 31):
            comb_val = e_z_val + alpha * gate_val * synergy_val
            acc = accuracy_score(y_val, (comb_val > 0.0).astype(np.int64))
            if acc > best_val_acc:
                best_val_acc = acc
                best_alpha = float(alpha)
                best_tau = float(tau)

    gate_te = np.exp(-0.5 * (e_z_te / best_tau) ** 2)
    comb_te = e_z_te + best_alpha * gate_te * synergy_te

    pred_f_only = (f_only_te > 0.0).astype(np.int64)
    pred_hybrid = (comb_te > 0.0).astype(np.int64)
    return pred_f_only, pred_hybrid, best_alpha

def bootstrap_paired_test(y_true, pred_base, pred_comp, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    acc_b = (pred_base == y_true).astype(np.float64) * 100.0
    acc_c = (pred_comp == y_true).astype(np.float64) * 100.0
    diff_trials = acc_c - acc_b
    mean_diff = float(np.mean(diff_trials))

    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    boot_means = np.mean(diff_trials[idx], axis=1)
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    _, p_val = ttest_rel(acc_c, acc_b)
    return mean_diff, ci_low, ci_high, float(p_val)

def calc_raw_generation_metrics(real_arr, synth_arr):
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
    r_task, s_task = real_test[:, :, 50:], synth_test[:, :, 50:]
    r_var, s_var = float(np.var(r_task)), float(np.var(s_task))
    r_peak = float(np.max(np.abs(r_task), axis=-1).mean())
    s_peak = float(np.max(np.abs(s_task), axis=-1).mean())
    r_lat = float((np.argmax(np.abs(r_task), axis=-1).mean() / 10.0))
    s_lat = float((np.argmax(np.abs(s_task), axis=-1).mean() / 10.0))

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

def evaluate_checkpoint(epoch="best", ddim_steps=50):
    with open("configs/shin_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    modality = cfg["data"]["target_modality"]
    processed_dir = cfg["data"]["processed_dir"]
    use_hrf = cfg["data"].get("eeg_hrf", True)

    ckpt_name = f"scdm_{modality.lower()}_best.pt" if str(epoch) == "best" else f"scdm_{modality.lower()}_ep{epoch}.pt"
    ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], ckpt_name)

    topo_dict = torch.load(os.path.join(processed_dir, f"topo_{modality.lower()}.pt"), map_location="cpu")
    topo_ef, topo_fe = topo_dict["topo_ef"].to(device), topo_dict["topo_fe"].to(device)
    corr_ef = topo_dict.get("corr_ef", None)
    spatial_affinity = topo_dict.get("spatial_affinity", None)
    if corr_ef is not None:
        corr_ef = corr_ef.to(device)
    if spatial_affinity is not None:
        spatial_affinity = spatial_affinity.to(device)

    unet = SCDMUnet(
        time_dim=cfg["model"].get("time_dim", 64),
        base_channels=cfg["model"].get("base_channels", 32),
        n_blocks=6
    ).to(device)
    model = SCDMDiffusion(unet, timesteps=cfg["model"]["timesteps"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    eeg_raw_t = torch.load(os.path.join(processed_dir, "eeg_epochs.pt"))
    eeg_hrf_t = torch.load(os.path.join(processed_dir, "eeg_hrf_epochs.pt"))
    eeg_cond = eeg_hrf_t if use_hrf else eeg_raw_t
    eeg_np = eeg_raw_t.numpy()

    real_target = torch.load(os.path.join(processed_dir, f"fnirs_{modality.lower()}_epochs.pt")).numpy()
    labels_np = torch.load(os.path.join(processed_dir, "labels.pt")).numpy()
    sub_ids_np = torch.load(os.path.join(processed_dir, "subject_ids.pt")).numpy()

    with open(os.path.join(processed_dir, "split_indices.json"), "r") as f:
        split_info = json.load(f)
    tr_idx = np.array(split_info["train_indices"], dtype=int)
    val_idx = np.array(split_info["val_indices"], dtype=int)
    te_idx = np.array(split_info["test_indices"], dtype=int)

    print(f">>> 正在使用 6-Block 标准 SCDM DDIM-{ddim_steps} 步采样生成合成 fNIRS (零测试集泄漏)...")
    synth_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(eeg_cond), 16), desc="DDIM 采样"):
            synth_list.append(
                model.sample(
                    eeg_cond[i:i + 16].to(device), topo_ef, topo_fe,
                    steps=ddim_steps, corr_ef=corr_ef, spatial_affinity=spatial_affinity, seed=5000 + i
                ).cpu().numpy()
            )
    synth_all = np.concatenate(synth_list, axis=0)

    v_mse, v_rmse, v_spcc, v_cpcc = calc_raw_generation_metrics(real_target[val_idx], synth_all[val_idx])
    t_mse, t_rmse, t_spcc, t_cpcc = calc_raw_generation_metrics(real_target[te_idx], synth_all[te_idx])
    left_idx, right_idx = get_motor_hemisphere_indices("datasets/raw")
    physio_stats = analyze_physiological_fidelity(real_target[te_idx], synth_all[te_idx], labels_np[te_idx], left_idx, right_idx)

    print("\n" + "=" * 88)
    print(f"【1. 方案 A 固定留出集信号生成质量与生理分布评估 (零测试集泄漏)】 (权重: {ckpt_name})")
    print(f"  [独立验证集 N={len(val_idx)}] MSE: {v_mse:.4f} (RMSE: {v_rmse:.4f}) | 单试次 PCC: {v_spcc:+.4f} | 曲线 PCC: {v_cpcc:+.4f}")
    print(f"  [留出测试集 N={len(te_idx)}] MSE: {t_mse:.4f} (RMSE: {t_rmse:.4f}) | 单试次 PCC: {t_spcc:+.4f} | 曲线 PCC: {t_cpcc:+.4f}")
    print(f"  [生理分布检验] 任务期方差: {physio_stats['var_ratio']} | 均值峰值: {physio_stats['peak_amp']} | 达峰时间: {physio_stats['peak_time_s']}")
    print(f"  [类别偏侧化度] 左/右运动区 LMI-RMI 不对称对比度: Real={physio_stats['hemi_contrast_real']:+.4f}, Synth={physio_stats['hemi_contrast_synth']:+.4f}")

    print("\n>>> 正在同一套固定划分 (Train=1218, Val=261, Test=261) 上运行 6 组对称分类对照实验 (5 Seeds)...")
    feat_real, real_c = extract_fnirs_motor_features(real_target, sub_ids_np, tr_idx, left_idx, right_idx)
    feat_synth, synth_c = extract_fnirs_motor_features(synth_all, sub_ids_np, tr_idx, left_idx, right_idx)

    rng_shuf = np.random.default_rng(42)
    tr_shuf = rng_shuf.permutation(tr_idx)
    val_shuf = rng_shuf.permutation(val_idx)
    te_shuf = rng_shuf.permutation(te_idx)

    modes = [
        ("eeg_only", "1. EEG-only (80Hz EEGNet)"),
        ("fnirs_real_only", "2. Real fNIRS-only (对照1)"),
        ("fnirs_synth_only", "3. Synth fNIRS-only (SCDM)"),
        ("eeg_shuffled_real", "4. EEG + Shuffled Real (对照2)"),
        ("eeg_real", "5. EEG + Real fNIRS (Val α*)"),
        ("eeg_synth", "6. EEG + Synth fNIRS (Val α*)")
    ]
    all_runs = {k: [] for k, _ in modes}
    pred_records = {k: [] for k, _ in modes}
    learned_alphas = {"real": [], "synth": [], "shuf": []}

    for run_i in range(5):
        seed_i = run_i * 100 + 42
        base_eeg, e_tr_lg, e_val_lg, e_te_lg = train_eeg_baseline(
            eeg_np[tr_idx], labels_np[tr_idx],
            eeg_np[val_idx], labels_np[val_idx],
            eeg_np[te_idx], device=device, seed=seed_i
        )
        p_eeg = ((e_te_lg[:, 1] - e_te_lg[:, 0]) > 0.0).astype(np.int64)

        p_f_real, p_hyb_real, a_real = train_modality_and_gated_fusion(
            base_eeg, e_tr_lg, e_val_lg, e_te_lg,
            eeg_np[tr_idx], eeg_np[val_idx], eeg_np[te_idx],
            real_c[tr_idx], real_c[val_idx], real_c[te_idx],
            feat_real[tr_idx], feat_real[val_idx], feat_real[te_idx],
            labels_np[tr_idx], labels_np[val_idx], device=device, seed=seed_i
        )
        p_f_syn, p_hyb_syn, a_syn = train_modality_and_gated_fusion(
            base_eeg, e_tr_lg, e_val_lg, e_te_lg,
            eeg_np[tr_idx], eeg_np[val_idx], eeg_np[te_idx],
            synth_c[tr_idx], synth_c[val_idx], synth_c[te_idx],
            feat_synth[tr_idx], feat_synth[val_idx], feat_synth[te_idx],
            labels_np[tr_idx], labels_np[val_idx], device=device, seed=seed_i
        )
        _, p_hyb_shuf, a_shuf = train_modality_and_gated_fusion(
            base_eeg, e_tr_lg, e_val_lg, e_te_lg,
            eeg_np[tr_idx], eeg_np[val_idx], eeg_np[te_idx],
            real_c[tr_shuf], real_c[val_shuf], real_c[te_shuf],
            feat_real[tr_shuf], feat_real[val_shuf], feat_real[te_shuf],
            labels_np[tr_idx], labels_np[val_idx], device=device, seed=seed_i
        )

        learned_alphas["real"].append(a_real)
        learned_alphas["synth"].append(a_syn)
        learned_alphas["shuf"].append(a_shuf)

        y_te_true = labels_np[te_idx]
        for k, p_arr in [
            ("eeg_only", p_eeg),
            ("fnirs_real_only", p_f_real),
            ("fnirs_synth_only", p_f_syn),
            ("eeg_shuffled_real", p_hyb_shuf),
            ("eeg_real", p_hyb_real),
            ("eeg_synth", p_hyb_syn)
        ]:
            all_runs[k].append(classification_metrics(y_te_true, p_arr))
            pred_records[k].append(p_arr)

    print("\n【2. 独立留出测试集 (N=261) 对照与多模态融合分类结果 (5-Seed Average)】")
    print(f"{'Experiment Condition':<30s} {'ACC':>12s} {'PRE':>10s} {'SEN':>10s} {'SPE':>10s}")
    print("-" * 88)
    for k, title in modes:
        avg = {m: np.mean([v[m] for v in all_runs[k]]) for m in all_runs[k][0]}
        std = {m: np.std([v[m] for v in all_runs[k]], ddof=1) for m in all_runs[k][0]}
        print(f"{title:<30s} {avg['ACC']:>6.2f}±{std['ACC']:>4.2f}% {avg['PRE']:>9.2f}% {avg['SEN']:>9.2f}% {avg['SPE']:>9.2f}%")

    y_te_true = labels_np[te_idx]
    ens_eeg = (np.mean(pred_records["eeg_only"], axis=0) >= 0.5).astype(np.int64)
    ens_real = (np.mean(pred_records["eeg_real"], axis=0) >= 0.5).astype(np.int64)
    ens_syn = (np.mean(pred_records["eeg_synth"], axis=0) >= 0.5).astype(np.int64)
    ens_shuf = (np.mean(pred_records["eeg_shuffled_real"], axis=0) >= 0.5).astype(np.int64)

    r_d, r_lo, r_hi, r_p = bootstrap_paired_test(y_te_true, ens_eeg, ens_real)
    s_d, s_lo, s_hi, s_p = bootstrap_paired_test(y_te_true, ens_eeg, ens_syn)
    sh_d, sh_lo, sh_hi, _ = bootstrap_paired_test(y_te_true, ens_eeg, ens_shuf)

    print("-" * 88)
    print(f"  [验证集选定平均门控权重 α*] Real α*={np.mean(learned_alphas['real']):.2f} | Synth α*={np.mean(learned_alphas['synth']):.2f} | Shuffled α*={np.mean(learned_alphas['shuf']):.2f}")
    print(f"  ★ 试次级配对检验 (EEG+Real  vs EEG-only, N=261): ΔACC = {r_d:+.2f}% | 95% Bootstrap CI: [{r_lo:+.2f}%, {r_hi:+.2f}%] | p = {r_p:.4f}")
    print(f"  ★ 试次级配对检验 (EEG+Synth vs EEG-only, N=261): ΔACC = {s_d:+.2f}% | 95% Bootstrap CI: [{s_lo:+.2f}%, {s_hi:+.2f}%] | p = {s_p:.4f}")
    print(f"  ★ 打乱配对对照   (EEG+Shuf  vs EEG-only, N=261): ΔACC = {sh_d:+.2f}% | 95% Bootstrap CI: [{sh_lo:+.2f}%, {sh_hi:+.2f}%]")
    print("=" * 88)

    time_axis = np.arange(256) / 10.0 - 5.0
    r_te, s_te = real_target[te_idx], synth_all[te_idx]
    plt.figure(figsize=(16, 4.5))
    for i, (m, t) in enumerate([
        (np.ones(len(te_idx), dtype=bool), f"Held-out Test Global (PCC: {t_cpcc:.4f})"),
        (y_te_true == 0, "Left Motor Imagery (LMI)"),
        (y_te_true == 1, "Right Motor Imagery (RMI)")
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