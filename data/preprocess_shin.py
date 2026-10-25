import os
import numpy as np
import scipy.io as sio
from scipy.signal import cheby2, butter, sosfiltfilt, resample, hilbert, fftconvolve
from scipy.special import gamma as gammafn
import torch
from tqdm import tqdm

def canonical_hrf(fs=160.0, duration=25.0):
    """Glover (1999) 标准双伽马血流动力学响应核 (并生成负向核匹配 HbR 脱氧血红蛋白下降特性)"""
    t = np.arange(0, duration, 1.0 / fs)
    a1, b1 = 6.0, 1.0
    a2, b2 = 16.0, 1.0
    c = 1.0 / 6.0
    h = ((t ** (a1 - 1) * np.exp(-t / b1)) / (b1 ** a1 * gammafn(a1))
         - c * (t ** (a2 - 1) * np.exp(-t / b2)) / (b2 ** a2 * gammafn(a2)))
    h[t < 0] = 0
    return (h / (np.abs(h).sum() + 1e-12)).astype(np.float32)

def eeg_to_hrf_features(eeg_trial, fs_eeg=160.0, n_out=256):
    """
    提取运动想象核心频段 (Mu: 8-13Hz, Beta: 13-30Hz, 全频带) 的 Hilbert 能量包络，
    并与标准 HRF 核进行卷积后降采样至 10Hz (30, 256)
    """
    n_ch, n_samp = eeg_trial.shape
    sos_mu = butter(4, [8.0, 13.0], btype='band', fs=fs_eeg, output='sos')
    sos_beta = butter(4, [13.0, 30.0], btype='band', fs=fs_eeg, output='sos')

    eeg_mu = sosfiltfilt(sos_mu, eeg_trial, axis=1)
    eeg_beta = sosfiltfilt(sos_beta, eeg_trial, axis=1)

    # 计算相对任务前 [-5s, 0s] (前800点) 的 ERD/ERS 功率变化包络
    pow_mu = np.abs(hilbert(eeg_mu, axis=1)) ** 2
    pow_mu = pow_mu - pow_mu[:, :800].mean(axis=1, keepdims=True)

    pow_beta = np.abs(hilbert(eeg_beta, axis=1)) ** 2
    pow_beta = pow_beta - pow_beta[:, :800].mean(axis=1, keepdims=True)

    pow_broad = np.abs(hilbert(eeg_trial, axis=1)) ** 2
    pow_broad = pow_broad - pow_broad[:, :800].mean(axis=1, keepdims=True)

    comb_pow = 0.45 * pow_mu + 0.45 * pow_beta + 0.10 * pow_broad
    hrf = canonical_hrf(fs_eeg)

    out = np.empty((n_ch, n_samp), dtype=np.float32)
    for ch in range(n_ch):
        out[ch] = fftconvolve(comb_pow[ch], hrf, mode='full')[:n_samp]

    res = resample(out, n_out, axis=1).astype(np.float32)
    # 保持任务前 [-5s, 0s] (前50点) 基线归零
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

    eeg_cnt_path = os.path.join(eeg_sub_dir, "with occular artifact", "cnt.mat")
    eeg_mrk_path = os.path.join(eeg_sub_dir, "with occular artifact", "mrk.mat")
    if not os.path.exists(eeg_cnt_path):
        eeg_cnt_path = os.path.join(eeg_sub_dir, "cnt.mat")
        eeg_mrk_path = os.path.join(eeg_sub_dir, "mrk.mat")

    fnirs_cnt_path = os.path.join(fnirs_sub_dir, "cnt.mat")
    fnirs_mrk_path = os.path.join(fnirs_sub_dir, "mrk.mat")

    eeg_mat = sio.loadmat(eeg_cnt_path, struct_as_record=False, squeeze_me=True)
    eeg_mrk_mat = sio.loadmat(eeg_mrk_path, struct_as_record=False, squeeze_me=True)
    fnirs_mat = sio.loadmat(fnirs_cnt_path, struct_as_record=False, squeeze_me=True)
    fnirs_mrk_mat = sio.loadmat(fnirs_mrk_path, struct_as_record=False, squeeze_me=True)

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
        od_low = f_raw[:, :36].T
        od_high = f_raw[:, 36:72].T
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

    all_eeg, all_hbo, all_hbr, all_labels, all_subs = [], [], [], [], []
    print(">>> [Step 1/3] 加载并预处理 Shin2017 数据集 (29 位被试)...")
    for sub in tqdm(range(1, 30), desc="被试处理进度"):
        e, hbo, hbr, l, s = process_single_subject(sub, raw_dir)
        all_eeg.append(e)
        all_hbo.append(hbo)
        all_hbr.append(hbr)
        all_labels.append(l)
        all_subs.append(s)

    eeg = np.concatenate(all_eeg, axis=0)
    hbo = np.concatenate(all_hbo, axis=0)
    hbr = np.concatenate(all_hbr, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subs = np.concatenate(all_subs, axis=0)

    print(">>> [Step 2/3] 执行 [-5s, 0s] 基线归零与跨通道联合标准化 (保留左右脑空间偏侧化对比度)...")
    for i in range(len(eeg)):
        mu = eeg[i, :, :800].mean(axis=1, keepdims=True)
        std = eeg[i].std(axis=1, keepdims=True)
        eeg[i] = (eeg[i] - mu) / np.where(std < 1e-8, 1.0, std)

        for arr in (hbo, hbr):
            # 1. 减去任务前 [-5s, 0s] (前50个采样点) 的静息基线，使信号在 0s 严格归零
            base_mu = arr[i, :, :50].mean(axis=1, keepdims=True)
            arr[i] = arr[i] - base_mu
            # 2. 使用试次内 36 通道联合标准差缩放 (保留激活通道与静息通道的相对幅值比！)
            trial_std = arr[i].std()
            arr[i] = arr[i] / (trial_std if trial_std > 1e-8 else 1.0)

    print(">>> [Step 3/3] 提取 Mu/Beta 频段 HRF 卷积特征 (1740, 30, 256)...")
    eeg_hrf = np.empty((len(eeg), 30, 256), dtype=np.float32)
    for i in tqdm(range(len(eeg)), desc="HRF 特征计算"):
        eeg_hrf[i] = eeg_to_hrf_features(eeg[i], fs_eeg=160.0, n_out=256)

    hrf_std = np.maximum(eeg_hrf.std(axis=(1, 2), keepdims=True), 1e-8)
    eeg_hrf = (eeg_hrf / hrf_std).astype(np.float32)

    torch.save(torch.from_numpy(eeg), os.path.join(save_dir, "eeg_epochs.pt"))
    torch.save(torch.from_numpy(eeg_hrf), os.path.join(save_dir, "eeg_hrf_epochs.pt"))
    torch.save(torch.from_numpy(hbo), os.path.join(save_dir, "fnirs_hbo_epochs.pt"))
    torch.save(torch.from_numpy(hbr), os.path.join(save_dir, "fnirs_hbr_epochs.pt"))
    torch.save(torch.from_numpy(labels), os.path.join(save_dir, "labels.pt"))
    torch.save(torch.from_numpy(subs), os.path.join(save_dir, "subject_ids.pt"))

    print("\n" + "=" * 60)
    print(">>> 预处理完成！基线与形状验证:")
    print(f"  - EEG 原始特征 : {eeg.shape}")
    print(f"  - EEG HRF 特征 : {eeg_hrf.shape}")
    print(f"  - HbR [-5s,0s] 基线均值: {hbr[:, :, :50].mean():.6f} (严格归零), 全局标准差: {hbr.std():.4f}")
    print(f"  - HbO [-5s,0s] 基线均值: {hbo[:, :, :50].mean():.6f} (严格归零), 全局标准差: {hbo.std():.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()