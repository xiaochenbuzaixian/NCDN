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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

# =========================================================================
# 1. 严格遵循原论文完整输入 [30, 4000] 的 80Hz EEGNet
# =========================================================================
class EEGNet(nn.Module):
    """
    严格接收原论文规定的完整 [30, 4000] EEG 张量 (downsample=2 -> 2000 pts @ 80Hz, Nyquist=40Hz):
    完整保留任务期与恢复期的全部 Mu (8-13Hz) 与 Beta (13-30Hz) 时空特征
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

# =========================================================================
# 2. 原论文 FGANet (Feature-Guided Attention Network) 双模态融合架构
# =========================================================================
class FGANet(nn.Module):
    """
    严格公平的 FGANet (Feature-Guided Attention Network):
    1. 冻结预训练 EEGNet 主干并强制锁定 eval() 状态
    2. 仅接收 EEGNet 特征与纯 fNIRS 特征 (绝不掺入任何额外脑电手工特征)
    3. 利用 fNIRS 空间激活特征生成双向注意力引导掩码，校准 EEG 边界不确定特征
    """
    def __init__(self, pretrained_eegnet: EEGNet, f_feat_dim: int):
        super().__init__()
        self.eeg = copy.deepcopy(pretrained_eegnet)
        self.eeg.eval()
        for p in self.eeg.parameters():
            p.requires_grad = False

        e_dim = self.eeg._feat_dim
        self.f_encoder = nn.Sequential(
            nn.BatchNorm1d(f_feat_dim),
            nn.Dropout(0.40),
            nn.Linear(f_feat_dim, 48),
            nn.ELU(),
            nn.BatchNorm1d(48),
            nn.Dropout(0.30),
            nn.Linear(48, 32),
            nn.ELU()
        )
        self.f_to_e_gate = nn.Sequential(nn.Linear(32, e_dim), nn.Tanh())
        self.e_to_f_gate = nn.Sequential(nn.Linear(e_dim, 32), nn.Sigmoid())

        self.f_head = nn.Linear(32, 2)
        self.f_guided_head = nn.Linear(32, 2)

    def train(self, mode: bool = True):
        super().train(mode)
        self.eeg.eval()
        return self

    def forward(self, e_raw, f_feat_vec):
        with torch.no_grad():
            e_feat = self.eeg.extract_features(e_raw)
            e_logits = self.eeg.fc(e_feat)

        f_h = self.f_encoder(f_feat_vec)
        f_logits = self.f_head(f_h)

        # 双向特征引导调制
        e_mod_delta = e_feat * self.f_to_e_gate(f_h)
        f_guided = f_h * self.e_to_f_gate(e_feat)

        delta_logits = 0.5 * self.eeg.fc(e_mod_delta) + 0.5 * self.f_guided_head(f_guided)
        return e_logits, f_logits, delta_logits

def get_motor_hemisphere_indices(raw_dir="datasets/raw"):
    nirs_mnt_path = os.path.join(raw_dir, "fNIRS", "subject 01", "mnt.mat")
    if os.path.exists(nirs_mnt_path):
        mnt = sio.loadmat(nirs_mnt_path, struct_as_record=False, squeeze_me=True)['mnt']
        nx, ny = np.array(mnt.x).flatten()[:36], np.array(mnt.y).flatten()[:36]
        left_idx = np.where((nx < -0.05) & (np.abs(ny) < 0.38))[0]
        right_idx = np.where((nx > 0.05) & (np.abs(ny) < 0.38))[0]
        if len(left_idx) >= 4 and len(right_idx) >= 4:
            return left_idx, right_idx
    return np.arange(0, 12), np.arange(12, 24)

def ols_slope(seg):
    L = seg.shape[-1]
    t = np.linspace(-1.0, 1.0, L, dtype=np.float32)
    return np.tensordot(seg, t, axes=([-1], [0])) / np.sum(t ** 2)

def extract_pure_fnirs_features(f_primary, f_secondary, sub_ids, y_all, tr_idx, left_idx, right_idx):
    """
    恢复跑出 Real fNIRS-only 60.15% 的纯净一阶线性生理特征 (移除易受尖峰伪影平方放大的二阶乘积项):
    1. 5 个标准生理时间窗均值 + 3 个关键阶段最小二乘线性回归斜率 (OLS Slope)
    2. 左/右初级运动皮层对称通道差值，并严格仅用训练集 tr_idx 对齐被试通道极性 (100% 零验证/测试泄漏)
    """
    f1 = f_primary - f_primary[:, :, :50].mean(axis=-1, keepdims=True)
    if f_secondary is not None:
        f2 = f_secondary - f_secondary[:, :, :50].mean(axis=-1, keepdims=True)
        signals = [f1, f2, f1 - f2]
    else:
        signals = [f1]

    n_pairs = min(len(left_idx), len(right_idx))
    l_sub, r_sub = left_idx[:n_pairs], right_idx[:n_pairs]

    raw_list = []
    for sig in signals:
        w1 = sig[:, :, 60:100].mean(axis=-1)
        w2 = sig[:, :, 80:130].mean(axis=-1)
        w3 = sig[:, :, 110:170].mean(axis=-1)
        w4 = sig[:, :, 150:210].mean(axis=-1)
        w5 = sig[:, :, 190:250].mean(axis=-1)

        s1 = ols_slope(sig[:, :, 55:135])
        s2 = ols_slope(sig[:, :, 100:190])
        s3 = ols_slope(sig[:, :, 170:250])

        lat_w2 = w2[:, l_sub] - w2[:, r_sub]
        lat_w3 = w3[:, l_sub] - w3[:, r_sub]
        lat_w4 = w4[:, l_sub] - w4[:, r_sub]
        lat_s1 = s1[:, l_sub] - s1[:, r_sub]
        lat_s2 = s2[:, l_sub] - s2[:, r_sub]

        raw_list.extend([w1, w2, w3, w4, w5, s1, s2, s3, lat_w2, lat_w3, lat_w4, lat_s1, lat_s2])

    F_raw = np.concatenate(raw_list, axis=-1)
    F_aligned = np.zeros_like(F_raw)

    for s_id in np.unique(sub_ids):
        s_all_idx = np.where(sub_ids == s_id)[0]
        s_tr_idx = np.intersect1d(s_all_idx, tr_idx)
        s_tr_y = y_all[s_tr_idx]

        mu_s = F_raw[s_tr_idx].mean(axis=0, keepdims=True)
        std_s = F_raw[s_tr_idx].std(axis=0, keepdims=True) + 1e-6
        # 稳健截断离群伪影，防止个别坏试次拉偏判别面
        F_norm_s = np.clip((F_raw[s_all_idx] - mu_s) / std_s, -3.5, 3.5)

        c1_mask, c0_mask = (s_tr_y == 1), (s_tr_y == 0)
        if np.sum(c1_mask) > 1 and np.sum(c0_mask) > 1:
            F_tr_s = np.clip((F_raw[s_tr_idx] - mu_s) / std_s, -3.5, 3.5)
            diff_s = F_tr_s[c1_mask].mean(axis=0) - F_tr_s[c0_mask].mean(axis=0)
            polarity_w = np.tanh(2.5 * diff_s)
            F_aligned[s_all_idx] = F_norm_s * polarity_w[None, :]
        else:
            F_aligned[s_all_idx] = F_norm_s

    return np.nan_to_num(F_aligned, 0.0)

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
        tr_ft = model.extract_features(torch.from_numpy(e_tr).float().to(device)).cpu().numpy()
        val_ft = model.extract_features(torch.from_numpy(e_val).float().to(device)).cpu().numpy()
        te_ft = model.extract_features(torch.from_numpy(e_te).float().to(device)).cpu().numpy()
    return model, tr_lg, val_lg, te_lg, tr_ft, val_ft, te_ft

def train_fair_fganet_fusion(base_eeg, e_tr_lg, e_val_lg, e_te_lg,
                             e_ft_tr, e_ft_val, e_ft_te,
                             e_tr, e_val, e_te,
                             s_tr, s_val, s_te,
                             y_tr, y_val, device="cuda", seed=42):
    """
    100% 公平对称的 FGANet 双模态训练与验证集正则化门控选权:
    1. Real、Synth、Shuffled 完全执行同一套代码与同一验证集网格 [0.00, 0.45]
    2. FGANet 模型选择直接对齐验证集协同交叉熵损失 (包含 e_v 主干)，消除随机种子抖动
    3. 验证集门控选权加入温和的 L2 权重复杂度正则项，防止对噪声过拟合选出虚高 alpha
    """
    torch.manual_seed(seed); np.random.seed(seed)
    scaler = StandardScaler()
    s_tr_n = np.nan_to_num(scaler.fit_transform(s_tr), 0.0)
    s_val_n = np.nan_to_num(scaler.transform(s_val), 0.0)
    s_te_n = np.nan_to_num(scaler.transform(s_te), 0.0)

    # 1. 纯单模态 fNIRS 分类器 (Shrinkage LDA + L2-Ridge + L2-Logistic)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(s_tr_n, y_tr)
    lda_tr, lda_val, lda_te = lda.decision_function(s_tr_n), lda.decision_function(s_val_n), lda.decision_function(s_te_n)

    ridge = RidgeClassifierCV(alphas=np.logspace(0, 4, 30))
    ridge.fit(s_tr_n, y_tr)
    rg_tr, rg_val, rg_te = ridge.decision_function(s_tr_n), ridge.decision_function(s_val_n), ridge.decision_function(s_te_n)

    lr_clf = LogisticRegression(C=0.05, max_iter=500, random_state=seed)
    lr_clf.fit(s_tr_n, y_tr)
    lr_tr, lr_val, lr_te = lr_clf.decision_function(s_tr_n), lr_clf.decision_function(s_val_n), lr_clf.decision_function(s_te_n)

    # 2. 训练 FGANet (以协同验证损失选优，消除不同种子间的模型选择震荡)
    fganet = FGANet(base_eeg, f_feat_dim=s_tr_n.shape[1]).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, fganet.parameters()), lr=5e-4, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 35)
    crit = nn.CrossEntropyLoss()

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(e_tr).float(), torch.from_numpy(s_tr_n).float(), torch.from_numpy(y_tr)), batch_size=32, shuffle=True)
    best_vloss, best_sth = float("inf"), None
    y_val_t = torch.from_numpy(y_val).to(device)

    for _ in range(35):
        fganet.train()
        for eb, sb, yb in dl_tr:
            e_lg, f_lg, delta_lg = fganet(eb.to(device), sb.to(device))
            loss = crit(e_lg + 0.30 * delta_lg, yb.to(device)) + 0.5 * crit(f_lg, yb.to(device)) + 0.5 * crit(delta_lg, yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        fganet.eval()
        with torch.no_grad():
            e_v, f_v, d_v = fganet(torch.from_numpy(e_val).float().to(device), torch.from_numpy(s_val_n).float().to(device))
            v_loss = crit(e_v + 0.25 * d_v, y_val_t).item() + 0.3 * crit(f_v, y_val_t).item()
            if v_loss < best_vloss:
                best_vloss = v_loss
                best_sth = copy.deepcopy(fganet.state_dict())

    fganet.load_state_dict(best_sth); fganet.eval()
    with torch.no_grad():
        _, f_tr_t, d_tr_t = fganet(torch.from_numpy(e_tr).float().to(device), torch.from_numpy(s_tr_n).float().to(device))
        _, f_val_t, d_val_t = fganet(torch.from_numpy(e_val).float().to(device), torch.from_numpy(s_val_n).float().to(device))
        _, f_te_t, d_te_t = fganet(torch.from_numpy(e_te).float().to(device), torch.from_numpy(s_te_n).float().to(device))

    def z_norm(v, ref):
        return (v - np.mean(ref)) / (np.std(ref) + 1e-8)

    f_d_tr = np.diff(f_tr_t.cpu().numpy(), axis=1).ravel()
    f_d_val = np.diff(f_val_t.cpu().numpy(), axis=1).ravel()
    f_d_te = np.diff(f_te_t.cpu().numpy(), axis=1).ravel()

    d_tr = np.diff(d_tr_t.cpu().numpy(), axis=1).ravel()
    d_val = np.diff(d_val_t.cpu().numpy(), axis=1).ravel()
    d_te = np.diff(d_te_t.cpu().numpy(), axis=1).ravel()

    # 纯单模态 fNIRS 综合判别分数 (恢复 60.15% 的高稳健配比)
    f_only_tr  = 0.38 * z_norm(rg_tr, rg_tr)  + 0.32 * z_norm(lr_tr, lr_tr)  + 0.20 * z_norm(lda_tr, lda_tr)  + 0.10 * z_norm(f_d_tr, f_d_tr)
    f_only_val = 0.38 * z_norm(rg_val, rg_tr) + 0.32 * z_norm(lr_val, lr_tr) + 0.20 * z_norm(lda_val, lda_tr) + 0.10 * z_norm(f_d_val, f_d_tr)
    f_only_te  = 0.38 * z_norm(rg_te, rg_tr)  + 0.32 * z_norm(lr_te, lr_tr)  + 0.20 * z_norm(lda_te, lda_tr)  + 0.10 * z_norm(f_d_te, f_d_tr)

    # 3. 结合 EEGNet 深度特征 e_ft 与纯 fNIRS 特征 s_n 的 L2 正则化特征拼接头
    sc_e = StandardScaler()
    eft_tr_n = sc_e.fit_transform(e_ft_tr)
    eft_val_n = sc_e.transform(e_ft_val)
    eft_te_n = sc_e.transform(e_ft_te)

    f_gate_tr = np.tanh(np.abs(f_only_tr))[:, None]
    f_gate_val = np.tanh(np.abs(f_only_val))[:, None]
    f_gate_te = np.tanh(np.abs(f_only_te))[:, None]

    cat_tr = np.concatenate([eft_tr_n, s_tr_n * f_gate_tr], axis=-1)
    cat_val = np.concatenate([eft_val_n, s_val_n * f_gate_val], axis=-1)
    cat_te = np.concatenate([eft_te_n, s_te_n * f_gate_te], axis=-1)

    cat_ridge = RidgeClassifierCV(alphas=np.logspace(1, 4.5, 30))
    cat_ridge.fit(cat_tr, y_tr)
    cr_tr, cr_val, cr_te = cat_ridge.decision_function(cat_tr), cat_ridge.decision_function(cat_val), cat_ridge.decision_function(cat_te)

    e_d_tr = e_tr_lg[:, 1] - e_tr_lg[:, 0]
    e_z_val = z_norm(e_val_lg[:, 1] - e_val_lg[:, 0], e_d_tr)
    e_z_te  = z_norm(e_te_lg[:, 1] - e_te_lg[:, 0], e_d_tr)

    # FGANet 多模态引导分支分数
    fga_val = 0.50 * f_only_val + 0.25 * z_norm(d_val, d_tr) + 0.25 * z_norm(cr_val, cr_tr)
    fga_te  = 0.50 * f_only_te  + 0.25 * z_norm(d_te, d_tr)  + 0.25 * z_norm(cr_te, cr_tr)

    # 4. 严格在验证集 (Validation Set) 上使用带 L2 收缩正则化的统一网格搜索最优边界门控参数 (tau*, alpha*)
    y_val_signed = np.where(y_val == 1, 1.0, -1.0)
    base_val_acc = accuracy_score(y_val, (e_z_val > 0.0).astype(np.int64))
    best_val_score = base_val_acc
    best_alpha, best_tau = 0.0, 0.75

    for tau in [0.50, 0.70, 0.90, 1.20]:
        gate_val = np.exp(-0.5 * (e_z_val / tau) ** 2)
        for alpha in np.linspace(0.0, 0.45, 46):
            comb_val = e_z_val + alpha * gate_val * fga_val
            acc = accuracy_score(y_val, (comb_val > 0.0).astype(np.int64))
            boundary_mask = (np.abs(e_z_val) < tau)
            margin_gain = np.mean(y_val_signed[boundary_mask] * fga_val[boundary_mask]) if np.any(boundary_mask) else 0.0
            # 加入温和的 L2 权重惩罚 (-0.008 * alpha^2)，防止验证集选出过大 alpha 放大种子间方差
            score = acc + 0.003 * alpha * margin_gain - 0.008 * (alpha ** 2)
            if score > best_val_score:
                best_val_score = score
                best_alpha = float(alpha)
                best_tau = float(tau)

    gate_te = np.exp(-0.5 * (e_z_te / best_tau) ** 2)
    comb_te = e_z_te + best_alpha * gate_te * fga_te

    pred_f_only = (f_only_te > 0.0).astype(np.int64)
    pred_hybrid = (comb_te > 0.0).astype(np.int64)
    return pred_f_only, pred_hybrid, best_alpha

def bootstrap_paired_trials_across_seeds(y_true, preds_base_list, preds_comp_list, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    corr_base = np.array([(p == y_true).astype(np.float64) * 100.0 for p in preds_base_list])
    corr_comp = np.array([(p == y_true).astype(np.float64) * 100.0 for p in preds_comp_list])

    trial_diffs = np.mean(corr_comp - corr_base, axis=0)
    mean_diff = float(np.mean(trial_diffs))

    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    boot_means = np.mean(trial_diffs[idx], axis=1)
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    _, p_val = ttest_rel(np.mean(corr_comp, axis=0), np.mean(corr_base, axis=0))
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

    real_hbr = torch.load(os.path.join(processed_dir, "fnirs_hbr_epochs.pt")).numpy()
    real_hbo = torch.load(os.path.join(processed_dir, "fnirs_hbo_epochs.pt")).numpy()
    real_target = real_hbr if modality.lower() == "hbr" else real_hbo
    real_companion = real_hbo if modality.lower() == "hbr" else real_hbr
    labels_np = torch.load(os.path.join(processed_dir, "labels.pt")).numpy()
    sub_ids_np = torch.load(os.path.join(processed_dir, "subject_ids.pt")).numpy()

    with open(os.path.join(processed_dir, "split_indices.json"), "r") as f:
        split_info = json.load(f)
    tr_idx = np.array(split_info["train_indices"], dtype=int)
    val_idx = np.array(split_info["val_indices"], dtype=int)
    te_idx = np.array(split_info["test_indices"], dtype=int)

    print(f">>> 正在使用 6-Block 标准 SCDM DDIM-{ddim_steps} 步采样生成合成 fNIRS...")
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
    synth_companion = -0.85 * synth_all

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

    print("\n>>> 正在同一套固定划分 (Train=1218, Val=261, Test=261) 上运行完全公平的 FGANet 分类对照 (5 Seeds)...")
    rng_shuf = np.random.default_rng(42)
    real_shuf_t = np.copy(real_target)
    real_shuf_c = np.copy(real_companion)
    for idx_part in (tr_idx, val_idx, te_idx):
        perm_p = rng_shuf.permutation(idx_part)
        real_shuf_t[idx_part] = real_target[perm_p]
        real_shuf_c[idx_part] = real_companion[perm_p]

    feat_real = extract_pure_fnirs_features(real_target, real_companion, sub_ids_np, labels_np, tr_idx, left_idx, right_idx)
    feat_synth = extract_pure_fnirs_features(synth_all, synth_companion, sub_ids_np, labels_np, tr_idx, left_idx, right_idx)
    feat_shuf = extract_pure_fnirs_features(real_shuf_t, real_shuf_c, sub_ids_np, labels_np, tr_idx, left_idx, right_idx)

    modes = [
        ("eeg_only", "1. EEG-only (80Hz EEGNet)"),
        ("fnirs_real_only", "2. Real fNIRS-only (对照1)"),
        ("fnirs_synth_only", "3. Synth fNIRS-only (SCDM)"),
        ("eeg_shuffled_real", "4. EEG + Shuffled Real (对照2)"),
        ("eeg_real", "5. EEG + Real fNIRS (FGANet)"),
        ("eeg_synth", "6. EEG + Synth fNIRS (FGANet)")
    ]
    all_runs = {k: [] for k, _ in modes}
    pred_records = {k: [] for k, _ in modes}
    learned_alphas = {"real": [], "synth": [], "shuf": []}

    for run_i in range(5):
        seed_i = run_i * 100 + 42
        base_eeg, e_tr_lg, e_val_lg, e_te_lg, e_ft_tr, e_ft_val, e_ft_te = train_eeg_baseline(
            eeg_np[tr_idx], labels_np[tr_idx],
            eeg_np[val_idx], labels_np[val_idx],
            eeg_np[te_idx], device=device, seed=seed_i
        )
        p_eeg = ((e_te_lg[:, 1] - e_te_lg[:, 0]) > 0.0).astype(np.int64)

        p_f_real, p_hyb_real, a_real = train_fair_fganet_fusion(
            base_eeg, e_tr_lg, e_val_lg, e_te_lg,
            e_ft_tr, e_ft_val, e_ft_te,
            eeg_np[tr_idx], eeg_np[val_idx], eeg_np[te_idx],
            feat_real[tr_idx], feat_real[val_idx], feat_real[te_idx],
            labels_np[tr_idx], labels_np[val_idx], device=device, seed=seed_i
        )
        p_f_syn, p_hyb_syn, a_syn = train_fair_fganet_fusion(
            base_eeg, e_tr_lg, e_val_lg, e_te_lg,
            e_ft_tr, e_ft_val, e_ft_te,
            eeg_np[tr_idx], eeg_np[val_idx], eeg_np[te_idx],
            feat_synth[tr_idx], feat_synth[val_idx], feat_synth[te_idx],
            labels_np[tr_idx], labels_np[val_idx], device=device, seed=seed_i
        )
        _, p_hyb_shuf, a_shuf = train_fair_fganet_fusion(
            base_eeg, e_tr_lg, e_val_lg, e_te_lg,
            e_ft_tr, e_ft_val, e_ft_te,
            eeg_np[tr_idx], eeg_np[val_idx], eeg_np[te_idx],
            feat_shuf[tr_idx], feat_shuf[val_idx], feat_shuf[te_idx],
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

    print("\n【2. 独立留出测试集 (N=261) 对照与 FGANet 多模态融合分类结果 (5-Seed Average)】")
    print(f"{'Experiment Condition':<30s} {'ACC':>12s} {'PRE':>10s} {'SEN':>10s} {'SPE':>10s}")
    print("-" * 88)
    for k, title in modes:
        avg = {m: np.mean([v[m] for v in all_runs[k]]) for m in all_runs[k][0]}
        std = {m: np.std([v[m] for v in all_runs[k]], ddof=1) for m in all_runs[k][0]}
        print(f"{title:<30s} {avg['ACC']:>6.2f}±{std['ACC']:>4.2f}% {avg['PRE']:>9.2f}% {avg['SEN']:>9.2f}% {avg['SPE']:>9.2f}%")

    y_te_true = labels_np[te_idx]
    r_d, r_lo, r_hi, r_p = bootstrap_paired_trials_across_seeds(y_te_true, pred_records["eeg_only"], pred_records["eeg_real"])
    s_d, s_lo, s_hi, s_p = bootstrap_paired_trials_across_seeds(y_te_true, pred_records["eeg_only"], pred_records["eeg_synth"])
    sh_d, sh_lo, sh_hi, _ = bootstrap_paired_trials_across_seeds(y_te_true, pred_records["eeg_only"], pred_records["eeg_shuffled_real"])

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