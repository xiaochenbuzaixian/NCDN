import os
import numpy as np
import scipy.io as sio
from scipy.signal import cheby2, butter, filtfilt, resample
import torch
from tqdm import tqdm

def bandpass_cheby2_eeg(data, fs=200.0, lowcut=0.5, highcut=50.0):
    """4阶 Chebyshev Type II 带通滤波器 (0.5 - 50 Hz)"""
    nyq = 0.5 * fs
    b, a = cheby2(4, rs=30, Wn=[lowcut / nyq, highcut / nyq], btype='bandpass')
    return filtfilt(b, a, data, axis=0)

def bandpass_butter_fnirs(data, fs=10.0, lowcut=0.01, highcut=0.1):
    """6阶零相位 Butterworth 带通滤波器 (0.01 - 0.1 Hz)"""
    nyq = 0.5 * fs
    b, a = butter(6, [lowcut / nyq, highcut / nyq], btype='bandpass')
    return filtfilt(b, a, data, axis=0)

def common_average_reference(eeg_data):
    """共平均参考 (CAR)"""
    return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

def mbll_convert(raw_fnirs):
    """
    将双波长原始光强信号 (72通道: 前36为760nm, 后36为850nm) 转换为 HbO 和 HbR (各36通道)
    若输入已为36通道，则直接返回
    """
    if raw_fnirs.shape[1] == 72:
        wl1 = np.maximum(raw_fnirs[:, :36], 1e-6)
        wl2 = np.maximum(raw_fnirs[:, 36:72], 1e-6)
        od1 = -np.log(wl1 / np.mean(wl1, axis=0, keepdims=True))
        od2 = -np.log(wl2 / np.mean(wl2, axis=0, keepdims=True))
        # 标准消光系数近似变换矩阵
        hbo = 1.4887 * od2 - 0.5970 * od1
        hbr = 1.8545 * od1 - 0.3315 * od2
        return hbo.astype(np.float32), hbr.astype(np.float32)
    else:
        ch = raw_fnirs[:, :36].astype(np.float32)
        return ch, ch

def extract_mi_sessions(cnt_obj, mrk_obj):
    """
    Shin2017 的 cnt/mrk 通常包含 6 个 session (其中 1, 3, 5 即索引 0, 2, 4 为 Motor Imagery)
    """
    if isinstance(cnt_obj, np.ndarray):
        if len(cnt_obj) == 6:
            indices = [0, 2, 4]
        else:
            indices = list(range(len(cnt_obj)))
        return [(cnt_obj[i], mrk_obj[i]) for i in indices]
    else:
        return [(cnt_obj, mrk_obj)]

def process_single_subject(sub_id, raw_dir):
    sub_str = f"subject {sub_id:02d}"
    eeg_sub_dir = os.path.join(raw_dir, "EEG", sub_str)
    fnirs_sub_dir = os.path.join(raw_dir, "fNIRS", sub_str)

    # 优先读取包含眼电的主任务文件 cnt.mat
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

        # 提取前30个脑电通道 (排除最后2个 EOG 通道)
        e_raw = cnt_e.x[:, :30].astype(np.float32)
        e_car = common_average_reference(e_raw)
        e_filt = bandpass_cheby2_eeg(e_car, fs=fs_e, lowcut=0.5, highcut=50.0)

        # 提取并转换 fNIRS (HbO & HbR)
        f_raw = cnt_f.x.astype(np.float32)
        hbo_raw, hbr_raw = mbll_convert(f_raw)
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

            # 截取窗口：任务开始前 5s 至开始后 20s (总长 25s / 25.6s)
            e_start = int((t_e_ms / 1000.0 - 5.0) * fs_e)
            e_end = e_start + int(25.0 * fs_e)

            f_start = int((t_f_ms / 1000.0 - 5.0) * fs_f)
            f_end = f_start + int(25.6 * fs_f)

            # 边界保护
            e_seg = np.pad(e_filt, ((max(0, -e_start), max(0, e_end - len(e_filt))), (0, 0)), mode='edge')
            e_seg = e_seg[max(0, e_start):max(0, e_start) + int(25.0 * fs_e)]

            hbo_seg = np.pad(hbo_filt, ((max(0, -f_start), max(0, f_end - len(hbo_filt))), (0, 0)), mode='edge')
            hbo_seg = hbo_seg[max(0, f_start):max(0, f_start) + int(25.6 * fs_f)]

            hbr_seg = np.pad(hbr_filt, ((max(0, -f_start), max(0, f_end - len(hbr_filt))), (0, 0)), mode='edge')
            hbr_seg = hbr_seg[max(0, f_start):max(0, f_start) + int(25.6 * fs_f)]

            # 重采样至论文标准长度: EEG 4000点 (160Hz), fNIRS 256点 (10Hz)
            e_res = resample(e_seg, 4000, axis=0).T       # [30, 4000]
            hbo_res = resample(hbo_seg, 256, axis=0).T    # [36, 256]
            hbr_res = resample(hbr_seg, 256, axis=0).T    # [36, 256]

            eeg_epochs.append(e_res)
            hbo_epochs.append(hbo_res)
            hbr_epochs.append(hbr_res)
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

    print(">>> 开始预处理 Shin2017 数据集 (29 位被试)...")
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

    # Z-score 标准化以稳定扩散模型训练
    all_eeg = (all_eeg - all_eeg.mean(dim=-1, keepdim=True)) / (all_eeg.std(dim=-1, keepdim=True) + 1e-6)
    all_hbo = (all_hbo - all_hbo.mean(dim=-1, keepdim=True)) / (all_hbo.std(dim=-1, keepdim=True) + 1e-6)
    all_hbr = (all_hbr - all_hbr.mean(dim=-1, keepdim=True)) / (all_hbr.std(dim=-1, keepdim=True) + 1e-6)

    print(f"\n>>> 处理完成！汇总形状:")
    print(f"  EEG:   {all_eeg.shape}  (期望: [1740, 30, 4000])")
    print(f"  HbO:   {all_hbo.shape}  (期望: [1740, 36, 256])")
    print(f"  HbR:   {all_hbr.shape}  (期望: [1740, 36, 256])")
    print(f"  Label: {all_labels.shape}")

    torch.save(all_eeg, os.path.join(save_dir, "eeg_epochs.pt"))
    torch.save(all_hbo, os.path.join(save_dir, "fnirs_hbo_epochs.pt"))
    torch.save(all_hbr, os.path.join(save_dir, "fnirs_hbr_epochs.pt"))
    torch.save(all_labels, os.path.join(save_dir, "labels.pt"))
    torch.save(all_subs, os.path.join(save_dir, "subject_ids.pt"))
    print(f">>> 所有张量已保存至 {save_dir}/")

if __name__ == "__main__":
    main()