import os
import json
import numpy as np
import scipy.io as sio
from scipy.signal import resample
import torch
import dcor
from tqdm import tqdm

def _nearest_free(r, c, used):
    for radius in range(1, 16):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 16 and 0 <= nc < 16 and (nr, nc) not in used:
                    return nr, nc
    raise RuntimeError("16x16 网格已满")

def load_real_montage(raw_dir="datasets/raw"):
    eeg_mnt_path = os.path.join(raw_dir, "EEG", "subject 01", "with occular artifact", "mnt.mat")
    if not os.path.exists(eeg_mnt_path):
        eeg_mnt_path = os.path.join(raw_dir, "EEG", "subject 01", "mnt.mat")
    nirs_mnt_path = os.path.join(raw_dir, "fNIRS", "subject 01", "mnt.mat")

    eeg_mnt = sio.loadmat(eeg_mnt_path, struct_as_record=False, squeeze_me=True)['mnt']
    nirs_mnt = sio.loadmat(nirs_mnt_path, struct_as_record=False, squeeze_me=True)['mnt']

    eeg_xy = np.column_stack([np.array(eeg_mnt.x).flatten()[:30], np.array(eeg_mnt.y).flatten()[:30]]).astype(np.float64)
    nirs_xy = np.column_stack([np.array(nirs_mnt.x).flatten()[:36], np.array(nirs_mnt.y).flatten()[:36]]).astype(np.float64)

    all_pos = np.vstack([eeg_xy, nirs_xy])
    mn, mx = all_pos.min(axis=0), all_pos.max(axis=0)
    span = np.where((mx - mn) == 0, 1.0, (mx - mn))
    norm = (all_pos - mn) / span
    cells = np.clip((norm * 15).round().astype(int), 0, 15)

    used, all_mapping = set(), {}
    for idx in range(66):
        r, c = int(cells[idx, 0]), int(cells[idx, 1])
        if (r, c) in used:
            r, c = _nearest_free(r, c, used)
        used.add((r, c))
        all_mapping[idx] = (r, c)

    eeg_coords = {i: all_mapping[i] for i in range(30)}
    fnirs_coords = {i: all_mapping[30 + i] for i in range(36)}

    eeg_norm, nirs_norm = norm[:30], norm[30:]
    dist_sq = np.sum((nirs_norm[:, None, :] - eeg_norm[None, :, :]) ** 2, axis=-1)
    spatial_affinity = np.exp(-dist_sq / (2 * 0.15 ** 2)).astype(np.float32)

    return eeg_coords, fnirs_coords, spatial_affinity, eeg_xy, nirs_xy

def distance_correlation_matrix(X, Y):
    Lx, Ly = X.shape[1], Y.shape[1]
    if Lx != Ly:
        L = min(Lx, Ly)
        if Lx != L:
            X = resample(X, L, axis=1)
        if Ly != L:
            Y = resample(Y, L, axis=1)
    Cx, Cy = X.shape[0], Y.shape[0]
    M = np.zeros((Cx, Cy), dtype=np.float32)
    for i in range(Cx):
        xi = np.ascontiguousarray(X[i], dtype=np.float64)
        for j in range(Cy):
            M[i, j] = dcor.distance_correlation(xi, np.ascontiguousarray(Y[j], dtype=np.float64))
    return M

def pearson_matrix(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = Xc @ Xc.T
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return (cov / (std[:, None] * std[None, :])).astype(np.float32)

def project_to_16x16(corr, coords16):
    Ca, Cb = corr.shape
    out = np.zeros((Ca, 16, 16), dtype=np.float32)
    for b in range(Cb):
        r, c = coords16[b]
        out[:, r, c] = corr[:, b]
    return out

def build_modality_planes(eeg_train, fnirs_train, eeg_coords, fnirs_coords, spatial_affinity, n_sample_trials=240):
    """严格仅在训练集样本 (eeg_train, fnirs_train) 上计算相关性矩阵，杜绝验证/测试集信息泄漏"""
    N_tr = len(eeg_train)
    idx = np.linspace(0, N_tr - 1, min(n_sample_trials, N_tr), dtype=int)
    cef_sum = np.zeros((30, 36), dtype=np.float64)
    ce_sum = np.zeros((30, 30), dtype=np.float64)
    cf_sum = np.zeros((36, 36), dtype=np.float64)

    for i in tqdm(idx, desc="仅在训练集上计算距离相关与皮尔逊平面"):
        e_i = resample(eeg_train[i], 256, axis=1)
        f_i = fnirs_train[i]
        cef_sum += distance_correlation_matrix(e_i, f_i)
        ce_sum += pearson_matrix(eeg_train[i])
        cf_sum += pearson_matrix(f_i)

    Cef = (cef_sum / len(idx)).astype(np.float32)
    Ce = (ce_sum / len(idx)).astype(np.float32)
    Cf = (cf_sum / len(idx)).astype(np.float32)

    topo_ef = torch.from_numpy(project_to_16x16(Cef, fnirs_coords))
    topo_fe = torch.from_numpy(project_to_16x16(Cef.T, eeg_coords))
    topo_ee = torch.from_numpy(project_to_16x16(Ce, eeg_coords))
    topo_ff = torch.from_numpy(project_to_16x16(Cf, fnirs_coords))

    return {
        "topo_ef": topo_ef,
        "topo_fe": topo_fe,
        "topo_ee": topo_ee,
        "topo_ff": topo_ff,
        "corr_ef": torch.from_numpy(Cef),
        "spatial_affinity": torch.from_numpy(spatial_affinity),
        "eeg_coords": eeg_coords,
        "fnirs_coords": fnirs_coords
    }

def main():
    processed_dir = "datasets/processed"
    eeg_coords, fnirs_coords, spatial_affinity, _, _ = load_real_montage("datasets/raw")
    print(f">>> 成功构建 66 通道共享 16x16 蒙太奇网格")

    # 严格读取原始 EEG (而非 HRF 特征) 以及固定的 train_indices
    eeg_all = torch.load(os.path.join(processed_dir, "eeg_epochs.pt")).numpy()
    with open(os.path.join(processed_dir, "split_indices.json"), "r") as f:
        train_idx = np.array(json.load(f)["train_indices"], dtype=int)

    eeg_train = eeg_all[train_idx]
    print(f">>> 严格限制在 {len(train_idx)} 个训练集试次上计算空间拓扑 (零验证/测试泄漏)")

    for mod in ["hbr", "hbo"]:
        fnirs_all = torch.load(os.path.join(processed_dir, f"fnirs_{mod}_epochs.pt")).numpy()
        fnirs_train = fnirs_all[train_idx]
        topo_dict = build_modality_planes(eeg_train, fnirs_train, eeg_coords, fnirs_coords, spatial_affinity)
        save_path = os.path.join(processed_dir, f"topo_{mod}.pt")
        torch.save(topo_dict, save_path)
        print(f">>> 已保存 {mod.upper()} 拓扑至 {save_path}")

if __name__ == "__main__":
    main()