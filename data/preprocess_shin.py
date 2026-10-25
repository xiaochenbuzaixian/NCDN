import os
import numpy as np
import scipy.io as sio
from scipy.signal import cheby2, butter, sosfiltfilt, resample
import torch
from tqdm import tqdm

# ==========================================
# 数值极稳定的 SOS 滤波
# ==========================================
def bandpass_cheby2_eeg(data, fs=200.0, lowcut=0.5, highcut=50.0):
    """EEG 滤波: 4阶 Chebyshev Type II (改用 SOS 格式彻底消除溢出)"""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    nyq = 0.5 * fs
    sos = cheby2(4, rs=30, Wn=[lowcut / nyq, highcut / nyq], btype='bandpass', output='sos')
    out = sosfiltfilt(sos, data, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def bandpass_butter_fnirs(data, fs=10.0, lowcut=0.01, highcut=0.1):
    """fNIRS 滤波: 4阶零相位 Butterworth (改用 SOS 格式防止极低频数值发散)"""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    nyq = 0.5 * fs
    # 4阶带通等效8阶，数值更稳且符合生理学频带
    sos = butter(4, [lowcut / nyq, highcut / nyq], btype='bandpass', output='sos')
    out = sosfiltfilt(sos, data, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def common_average_reference(eeg_data):
    """共平均参考 (CAR)"""
    return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

def parse_fnirs_signals(raw_fnirs):
    """
    自适应解析 fNIRS:
    1. 若为 72 通道双波长光强，使用 MBLL 转换为 HbO 与 HbR
    2. 若已为 36 通道，则直接分离或使用
    """
    raw_fnirs = np.nan_to_num(raw_fnirs, nan=0.0, posinf=0.0, neginf=0.0)
    if raw_fnirs.shape[1] == 72:
        wl1 = np.maximum(raw_fnirs[:, :36], 1e-4)
        wl2 = np.maximum(raw_fnirs[:, 36:72], 1e-4)
        mean_wl1 = np.maximum(np.mean(wl1, axis=0, keepdims=True), 1e-4)
        mean_wl2 = np.maximum(np.mean(wl2, axis=0, keepdims=True), 1e-4)
        od1 = -np.log(wl1 / mean_wl1)
        od2 = -np.log(wl2 / mean_wl2)
        hbo = 1.4887 * od2 - 0.5970 * od1
        hbr = 1.8545 * od1 - 0.3315 * od2
        return np.nan_to_num(hbo), np.nan_to_num(hbr)
    else:
        # 已为 36 通道时直接使用
        ch = raw_fnirs[:, :36]
        return np.nan_to_num(ch), np.nan_to_num(-ch)

def extract_mi_sessions(cnt_obj, mrk_obj):
    if isinstance(cnt_obj, np.ndarray):
        indices = [0, 2, 4] if len(cnt_obj) == 6 else list(range(len(cnt_obj)))
        return [(cnt_obj[i], mrk_obj[i]) for i in indices]
    else:
        return [(cnt_obj, mrk_obj)]

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

    eeg_epochs, hbo_epochs, hbr_epochs, trial_labels, sub_ids = [], [], [], [], []

    for (cnt_e, mrk_e), (cnt_f, mrk_f) in zip(eeg_sessions, fnirs_sessions):
        fs_e = float(cnt_e.fs)
        fs_f = float(cnt_f.fs)

        e_raw = cnt_e.x[:, :30].astype(np.float64)
        e_car = common_average_reference(e_raw)
        e_filt = bandpass_cheby2_eeg(e_car, fs=fs_e, lowcut=0.5, highcut=50.0)

        f_raw = cnt_f.x.astype(np.float64)
        hbo_raw, hbr_raw = parse_fnirs_signals(f_raw)
        hbo_filt = bandpass_butter_fnirs(hbo_raw, fs=fs_f, lowcut=0.01, highcut=0.1)
        hbr_filt = bandpass_butter_fnirs(hbr_raw, fs=fs_f, lowcut=0.01, highcut=0.1)

        times_e = mrk_e.time
        times_f = mrk_f.time
        y = mrk_e.y
        if y.ndim > 1:
            y = np.argmax(y, axis=0)

        for idx in range(len(times_e)):
            t_e_ms = times_e[idx]
            t_f_ms = times_f[idx]

            e_start = int((t_e_ms / 1000.0 - 5.0) * fs_e)
            e_end = e_start + int(25.0 * fs_e)

            f_start = int((t_f_ms / 1000.0 - 5.0) * fs_f)
            f_end = f_start + int(25.6 * fs_f)

            e_seg = np.pad(e_filt, ((max(0, -e_start), max(0, e_end - len(e_filt))), (0, 0)), mode='edge')
            e_seg = e_seg[max(0, e_start):max(0, e_start) + int(25.0 * fs_e)]

            hbo_seg = np.pad(hbo_filt, ((max(0, -f_start), max(0, f_end - len(hbo_filt))), (0, 0)), mode='edge')
            hbo_seg = hbo_seg[max(0, f_start):max(0, f_start) + int(25.6 * fs_f)]

            hbr_seg = np.pad(hbr_filt, ((max(0, -f_start), max(0, f_end - len(hbr_filt))), (0, 0)), mode='edge')
            hbr_seg = hbr_seg[max(0, f_start):max(0, f_start) + int(25.6 * fs_f)]

            e_res = resample(e_seg, 4000, axis=0).T       # [30, 4000]
            hbo_res = resample(hbo_seg, 256, axis=0).T    # [36, 256]
            hbr_res = resample(hbr_seg, 256, axis=0).T    # [36, 256]

            eeg_epochs.append(e_res.astype(np.float32))
            hbo_epochs.append(hbo_res.astype(np.float32))
            hbr_epochs.append(hbr_res.astype(np.float32))
            trial_labels.append(int(y[idx]))
            sub_ids.append(sub_id - 1)

    return (np.array(eeg_epochs, dtype=np.float32),
            np.array(hbo_epochs, dtype=np.float32),
            np.array(hbr_epochs, dtype=np.float32),
            np.array(trial_labels, dtype=np.int64),
            np.array(sub_ids, dtype=np.int64))

def main():
    raw_dir = "datasets/raw"
    save_dir = "datasets/processed"
    os.makedirs(save_dir, exist_ok=True)

    all_eeg, all_hbo, all_hbr, all_labels, all_subs = [], [], [], [], []

    print(">>> 开始预处理 Shin2017 数据集 (使用 SOS 稳定滤波)...")
    for sub in tqdm(range(1, 30)):
        e, hbo, hbr, l, s = process_single_subject(sub, raw_dir)
        all_eeg.append(e)
        all_hbo.append(hbo)
        all_hbr.append(hbr)
        all_labels.append(l)
        all_subs.append(s)

    all_eeg = torch.tensor(np.concatenate(all_eeg, axis=0))
    all_hbo = torch.tensor(np.concatenate(all_hbo, axis=0))
    all_hbr = torch.tensor(np.concatenate(all_hbr, axis=0))
    all_labels = torch.tensor(np.concatenate(all_labels, axis=0))
    all_subs = torch.tensor(np.concatenate(all_subs, axis=0))

    # 安全 Z-Score 标准化
    def safe_zscore(tensor):
        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True)
        std = torch.where(std < 1e-4, torch.ones_like(std), std)
        normed = (tensor - mean) / std
        return torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    all_eeg = safe_zscore(all_eeg)
    all_hbo = safe_zscore(all_hbo)
    all_hbr = safe_zscore(all_hbr)

    print(f"\n>>> 处理完成！形状汇总:")
    print(f"  EEG:   {all_eeg.shape}  | NaN数: {torch.isnan(all_eeg).sum().item()}")
    print(f"  HbO:   {all_hbo.shape}  | NaN数: {torch.isnan(all_hbo).sum().item()}")
    print(f"  HbR:   {all_hbr.shape}  | NaN数: {torch.isnan(all_hbr).sum().item()}")

    torch.save(all_eeg, os.path.join(save_dir, "eeg_epochs.pt"))
    torch.save(all_hbo, os.path.join(save_dir, "fnirs_hbo_epochs.pt"))
    torch.save(all_hbr, os.path.join(save_dir, "fnirs_hbr_epochs.pt"))
    torch.save(all_labels, os.path.join(save_dir, "labels.pt"))
    torch.save(all_subs, os.path.join(save_dir, "subject_ids.pt"))
    print(f">>> 纯净张量已保存至 {save_dir}/")

if __name__ == "__main__":
    main()