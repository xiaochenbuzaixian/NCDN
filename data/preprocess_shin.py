import io
import os
import json
import numpy as np
import scipy.io as sio
from scipy.signal import cheby2, butter, sosfiltfilt, resample, hilbert, fftconvolve
from scipy.special import gamma as gammafn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
from tqdm import tqdm

def safe_loadmat(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
        return sio.loadmat(io.BytesIO(raw_bytes), struct_as_record=False, squeeze_me=True)
    except Exception as e:
        raise OSError(f"文件损坏或未传输完整: {path} (当前大小: {file_size_mb:.2f} MB) | 原始错误: {e}")

def load_subject_mat_with_fallback(sub_dir, filename):
    candidates = [
        os.path.join(sub_dir, "with occular artifact", filename),
        os.path.join(sub_dir, filename),
        os.path.join(sub_dir, "without occular artifact", filename),
    ]
    last_err = None
    for p in candidates:
        if os.path.exists(p):
            try:
                return safe_loadmat(p)
            except Exception as e:
                last_err = e
    raise RuntimeError(f"被试目录 {sub_dir} 下的 {filename} 无法读取: {last_err}")

def canonical_hrf_with_derivative(fs=160.0, duration=25.0):
    t = np.arange(0, duration, 1.0 / fs)
    a1, b1 = 6.0, 1.0
    a2, b2 = 16.0, 1.0
    c = 1.0 / 6.0
    h = ((t ** (a1 - 1) * np.exp(-t / b1)) / (b1 ** a1 * gammafn(a1))
         - c * (t ** (a2 - 1) * np.exp(-t / b2)) / (b2 ** a2 * gammafn(a2)))
    h[t < 0] = 0
    h = (h / (np.abs(h).sum() + 1e-12)).astype(np.float32)
    dh = np.gradient(h)
    dh = (dh / (np.abs(dh).sum() + 1e-12)).astype(np.float32)
    return h, dh

def eeg_to_hrf_features(eeg_trial, fs_eeg=160.0, n_out=256):
    """提取 Mu (8-13Hz) 与 Beta (13-30Hz) 相对任务前基线 [-5s, 0s] 的 ERD/ERS 功率包络并卷积双基 HRF"""
    n_ch, n_samp = eeg_trial.shape
    sos_mu = butter(4, [8.0, 13.0], btype='band', fs=fs_eeg, output='sos')
    sos_beta = butter(4, [13.0, 30.0], btype='band', fs=fs_eeg, output='sos')
    eeg_mu = sosfiltfilt(sos_mu, eeg_trial, axis=1)
    eeg_beta = sosfiltfilt(sos_beta, eeg_trial, axis=1)

    pow_mu = np.log(np.abs(hilbert(eeg_mu, axis=1)) ** 2 + 1e-6)
    pow_mu = pow_mu - pow_mu[:, :800].mean(axis=1, keepdims=True)
    pow_beta = np.log(np.abs(hilbert(eeg_beta, axis=1)) ** 2 + 1e-6)
    pow_beta = pow_beta - pow_beta[:, :800].mean(axis=1, keepdims=True)

    neural_drive = -(0.55 * pow_mu + 0.45 * pow_beta)
    h, dh = canonical_hrf_with_derivative(fs_eeg)
    out = np.empty((n_ch, n_samp), dtype=np.float32)
    for ch in range(n_ch):
        c_main = fftconvolve(neural_drive[ch], h, mode='full')[:n_samp]
        c_deriv = fftconvolve(neural_drive[ch], dh, mode='full')[:n_samp]
        out[ch] = c_main + 0.3 * c_deriv
    res = resample(out, n_out, axis=1).astype(np.float32)
    # 严格对齐任务前 [-5s, 0s] (前50点) 基线归零，保留试次间与通道间的真实幅值差异
    res = res - res[:, :50].mean(axis=1, keepdims=True)
    return res

def mbll_from_od(od_low, od_high, dpf_low=6.26, dpf_high=5.57, d=3.0):
    e_hbo_760, e_hbr_760 = 1486.5865, 3843.707
    e_hbo_850, e_hbr_850 = 2526.391, 1798.643
    E = np.array([
        [e_hbo_760 * dpf_low * d, e_hbr_760 * dpf_low * d],
        [e_hbo_850 * dpf_high * d, e_hbr_850 * dpf_high * d]
    ], dtype=np.float64)
    E_inv = np.linalg.inv(E)
    dod_low = od_low - od_low.mean(axis=1, keepdims=True)
    dod_high = od_high - od_high.mean(axis=1, keepdims=True)
    n_ch, n_t = od_low.shape
    hbo = np.zeros((n_ch, n_t), dtype=np.float64)
    hbr = np.zeros((n_ch, n_t), dtype=np.float64)
    for ch in range(n_ch):
        dod = np.vstack([dod_low[ch:ch+1, :], dod_high[ch:ch+1, :]])
        conc = E_inv @ dod
        hbo[ch] = conc[0]
        hbr[ch] = conc[1]
    return hbo, hbr

def preprocess_eeg_continuous(eeg_raw, fs_in=200.0, fs_out=160.0):
    x = eeg_raw - eeg_raw.mean(axis=0, keepdims=True)
    sos = cheby2(4, 40, [0.5, 50], btype="band", fs=fs_in, output="sos")
    x = sosfiltfilt(sos, x, axis=1)
    n_out = int(round(x.shape[1] * fs_out / fs_in))
    return resample(x, n_out, axis=1).astype(np.float32)

def extract_mi_sessions(cnt_obj, mrk_obj):
    if isinstance(cnt_obj, np.ndarray):
        indices = [0, 2, 4] if len(cnt_obj) == 6 else list(range(len(cnt_obj)))
        return [(cnt_obj[i], mrk_obj[i]) for i in indices]
    return [(cnt_obj, mrk_obj)]

def parse_trial_labels(mrk):
    if hasattr(mrk, 'event') and hasattr(mrk.event, 'desc'):
        desc = np.array(mrk.event.desc).flatten().astype(int)
        if np.any(desc == 16):
            return np.where(desc == 16, 0, 1).astype(np.int64)
        return (desc - 1).astype(np.int64)
    y = np.array(mrk.y)
    if y.ndim > 1:
        return np.argmax(y, axis=0).astype(np.int64)
    return y.astype(np.int64)

def process_single_subject(sub_id, raw_dir):
    sub_str = f"subject {sub_id:02d}"
    eeg_sub_dir = os.path.join(raw_dir, "EEG", sub_str)
    fnirs_sub_dir = os.path.join(raw_dir, "fNIRS", sub_str)
    eeg_mat = load_subject_mat_with_fallback(eeg_sub_dir, "cnt.mat")
    eeg_mrk_mat = load_subject_mat_with_fallback(eeg_sub_dir, "mrk.mat")
    fnirs_mat = load_subject_mat_with_fallback(fnirs_sub_dir, "cnt.mat")
    fnirs_mrk_mat = load_subject_mat_with_fallback(fnirs_sub_dir, "mrk.mat")

    eeg_sessions = extract_mi_sessions(eeg_mat['cnt'], eeg_mrk_mat['mrk'])
    fnirs_sessions = extract_mi_sessions(fnirs_mat['cnt'], fnirs_mrk_mat['mrk'])
    sos_fnirs = butter(6, [0.01, 0.1], btype='band', fs=10.0, output='sos')
    eeg_epochs, hbo_epochs, hbr_epochs, trial_labels, sub_ids = [], [], [], [], []

    for (cnt_e, mrk_e), (cnt_f, mrk_f) in zip(eeg_sessions, fnirs_sessions):
        fs_e = float(cnt_e.fs)
        fs_f = float(cnt_f.fs)
        if hasattr(cnt_e, 'clab'):
            clab = [str(c) for c in np.array(cnt_e.clab).flatten()]
            eeg_idx = [i for i, name in enumerate(clab) if name not in ('VEOG', 'HEOG')][:30]
        else:
            eeg_idx = list(range(30))
        e_raw = np.array(cnt_e.x[:, eeg_idx], dtype=np.float64).T
        e_proc = preprocess_eeg_continuous(e_raw, fs_in=fs_e, fs_out=160.0)
        f_raw = np.array(cnt_f.x, dtype=np.float64)
        od_low, od_high = f_raw[:, :36].T, f_raw[:, 36:72].T
        hbo_cont, hbr_cont = mbll_from_od(od_low, od_high)
        hbo_filt = sosfiltfilt(sos_fnirs, hbo_cont, axis=1)
        hbr_filt = sosfiltfilt(sos_fnirs, hbr_cont, axis=1)
        if abs(fs_f - 10.0) > 0.1:
            n_out_f = int(round(hbo_filt.shape[1] * 10.0 / fs_f))
            hbo_filt = resample(hbo_filt, n_out_f, axis=1)
            hbr_filt = resample(hbr_filt, n_out_f, axis=1)
        times_e = np.array(mrk_e.time).flatten().astype(np.float64)
        times_f = np.array(mrk_f.time).flatten().astype(np.float64)
        labels = parse_trial_labels(mrk_e)
        for idx in range(len(times_e)):
            e_onset = int(round(times_e[idx] / 1000.0 * 160.0))
            e_start = max(0, min(e_onset - int(5.0 * 160.0), e_proc.shape[1] - 4000))
            e_trial = e_proc[:, e_start:e_start + 4000]
            f_onset = int(round(times_f[idx] / 1000.0 * 10.0))
            f_start = max(0, min(f_onset - int(5.0 * 10.0), hbo_filt.shape[1] - 256))
            hbo_trial = hbo_filt[:, f_start:f_start + 256]
            hbr_trial = hbr_filt[:, f_start:f_start + 256]
            eeg_epochs.append(e_trial.astype(np.float32))
            hbo_epochs.append(np.clip(hbo_trial, -1e10, 1e10).astype(np.float32))
            hbr_epochs.append(np.clip(hbr_trial, -1e10, 1e10).astype(np.float32))
            trial_labels.append(int(labels[idx]))
            sub_ids.append(sub_id - 1)
    return (np.stack(eeg_epochs, axis=0),
            np.stack(hbo_epochs, axis=0),
            np.stack(hbr_epochs, axis=0),
            np.array(trial_labels, dtype=np.int64),
            np.array(sub_ids, dtype=np.int64))

def main():
    raw_dir = "datasets/raw"
    save_dir = "datasets/processed"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    all_eeg, all_hbo, all_hbr, all_labels, all_subs = [], [], [], [], []

    print(">>> [Step 1/4] 加载并预处理 Shin2017 数据集 (29 位被试)...")
    for sub in tqdm(range(1, 30), desc="被试处理进度"):
        e, hbo, hbr, l, s = process_single_subject(sub, raw_dir)
        all_eeg.append(e); all_hbo.append(hbo); all_hbr.append(hbr)
        all_labels.append(l); all_subs.append(s)

    eeg = np.concatenate(all_eeg, axis=0)
    hbo = np.concatenate(all_hbo, axis=0)
    hbr = np.concatenate(all_hbr, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subs = np.concatenate(all_subs, axis=0)

    # 生成全局唯一、严格分层的数据集划分索引 (70% 训练, 15% 验证, 15% 留出测试 + 真正的 5-Fold 交叉验证)
    print(">>> [Step 2/4] 构建全局统一的训练/验证/测试集与严格 5 折交叉验证划分 (消除数据泄漏)...")
    strata = subs * 2 + labels  # 按 (被试, 类别) 联合分层
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(eeg, strata))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel, test_rel = next(sss2.split(eeg[temp_idx], strata[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = []
    for f_tr, f_te in skf.split(eeg, strata):
        cv_folds.append({"train": f_tr.tolist(), "test": f_te.tolist()})

    split_info = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "cv_5fold": cv_folds
    }
    with open(os.path.join(save_dir, "split_indices.json"), "w") as f:
        json.dump(split_info, f)

    print(">>> [Step 3/4] 执行任务前基线归零与基于训练集统计量的标准化 (保留试次间与通道间幅值信息)...")
    for i in range(len(eeg)):
        mu = eeg[i, :, :800].mean(axis=1, keepdims=True)
        std = eeg[i].std(axis=1, keepdims=True)
        eeg[i] = (eeg[i] - mu) / np.where(std < 1e-8, 1.0, std)

    for arr in (hbo, hbr):
        # 1. 仅减去任务前 [-5s, 0s] (前50点) 静息基线，绝不减去全时段均值，完整保留任务期响应幅值与偏侧化！
        arr -= arr[:, :, :50].mean(axis=-1, keepdims=True)
        # 2. 按每个被试在训练集内的标准差归一化，避免抹平试次间与通道间的运动想象差异
        for s_id in np.unique(subs):
            s_mask = (subs == s_id)
            s_tr_mask = np.intersect1d(np.where(s_mask)[0], train_idx)
            scale_ref = arr[s_tr_mask].std() if len(s_tr_mask) > 0 else arr[s_mask].std()
            arr[s_mask] /= max(float(scale_ref), 1e-8)

    print(f">>> [Step 4/4] 提取神经血管耦合特征 ({len(eeg)}, 30, 256)...")
    eeg_hrf = np.empty((len(eeg), 30, 256), dtype=np.float32)
    for i in tqdm(range(len(eeg)), desc="HRF 特征计算"):
        eeg_hrf[i] = eeg_to_hrf_features(eeg[i], fs_eeg=160.0, n_out=256)
    hrf_train_std = eeg_hrf[train_idx].std(axis=(0, 2), keepdims=True)
    eeg_hrf = eeg_hrf / np.maximum(hrf_train_std, 1e-6)

    torch.save(torch.from_numpy(eeg), os.path.join(save_dir, "eeg_epochs.pt"))
    torch.save(torch.from_numpy(eeg_hrf), os.path.join(save_dir, "eeg_hrf_epochs.pt"))
    torch.save(torch.from_numpy(hbo), os.path.join(save_dir, "fnirs_hbo_epochs.pt"))
    torch.save(torch.from_numpy(hbr), os.path.join(save_dir, "fnirs_hbr_epochs.pt"))
    torch.save(torch.from_numpy(labels), os.path.join(save_dir, "labels.pt"))
    torch.save(torch.from_numpy(subs), os.path.join(save_dir, "subject_ids.pt"))

    print("\n" + "=" * 65)
    print(">>> 预处理完成！严格数据划分与统计验证:")
    print(f"  - 训练集: {len(train_idx)} | 验证集: {len(val_idx)} | 留出测试集: {len(test_idx)}")
    print(f"  - HbR [-5s,0s] 基线均值: {hbr[:, :, :50].mean():.6f} | 任务期 std: {hbr[:, :, 50:].std():.4f}")
    print(f"  - HbO [-5s,0s] 基线均值: {hbo[:, :, :50].mean():.6f} | 任务期 std: {hbo[:, :, 50:].std():.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()