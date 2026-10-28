import os
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
    """读取 subject 01 的 mnt.mat，在共享坐标系下构建 66 通道无碰撞 16x16 网格映射及物理距离矩阵"""
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
    # 计算 36x30 的真实头皮欧氏距离高斯亲和矩阵 (作为解剖学空间先验)
    eeg_norm, nirs_norm = norm[:30], norm[30:]
    dist_sq = np.sum((nirs_norm[:, None, :] - eeg_norm[None, :, :]) ** 2, axis=-1)
    spatial_affinity = np.exp(-dist_sq / (2 * 0.15 ** 2)).astype(np.float32)  # (36, 30)
    return eeg_coords, fnirs_coords, spatial_affinity

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

def build_modality_planes(eeg_all, fnirs_all, eeg_coords, fnirs_coords, spatial_affinity, n_sample_trials=200):
    N = len(eeg_all)
    idx = np.linspace(0, int(N * 0.8) - 1, min(n_sample_trials, int(N * 0.8)), dtype=int)
    cef_sum = np.zeros((30, 36), dtype=np.float64)
    ce_sum = np.zeros((30, 30), dtype=np.float64)
    cf_sum = np.zeros((36, 36), dtype=np.float64)
    for i in tqdm(idx, desc="计算距离相关与皮尔逊平面"):
        e_i = resample(eeg_all[i], 128, axis=1)
        f_i = resample(fnirs_all[i], 128, axis=1)
        cef_sum += distance_correlation_matrix(e_i, f_i)
        ce_sum += pearson_matrix(eeg_all[i])
        cf_sum += pearson_matrix(fnirs_all[i])
    Cef = (cef_sum / len(idx)).astype(np.float32)  # (30, 36)
    Ce = (ce_sum / len(idx)).astype(np.float32)    # (30, 30)
    Cf = (cf_sum / len(idx)).astype(np.float32)    # (36, 36)
    topo_ef = torch.from_numpy(project_to_16x16(Cef, fnirs_coords))     # (30, 16, 16)
    topo_fe = torch.from_numpy(project_to_16x16(Cef.T, eeg_coords))     # (36, 16, 16)
    topo_ee = torch.from_numpy(project_to_16x16(Ce, eeg_coords))        # (30, 16, 16)
    topo_ff = torch.from_numpy(project_to_16x16(Cf, fnirs_coords))      # (36, 16, 16)
    return {
        "topo_ef": topo_ef,
        "topo_fe": topo_fe,
        "topo_ee": topo_ee,
        "topo_ff": topo_ff,
        "corr_ef": torch.from_numpy(Cef),                    # (30, 36) 直接跨模态相关矩阵
        "spatial_affinity": torch.from_numpy(spatial_affinity), # (36, 30) 物理导联场亲和矩阵
        "eeg_coords": eeg_coords,
        "fnirs_coords": fnirs_coords
    }

def main():
    processed_dir = "datasets/processed"
    eeg_coords, fnirs_coords, spatial_affinity = load_real_montage("datasets/raw")
    print(f">>> 成功构建 66 通道共享 16x16 蒙太奇网格 (无碰撞单元数: {len(eeg_coords) + len(fnirs_coords)})")
    eeg_hrf_path = os.path.join(processed_dir, "eeg_hrf_epochs.pt")
    eeg_path = eeg_hrf_path if os.path.exists(eeg_hrf_path) else os.path.join(processed_dir, "eeg_epochs.pt")
    eeg_all = torch.load(eeg_path).numpy()
    for mod in ["hbr", "hbo"]:
        fnirs_all = torch.load(os.path.join(processed_dir, f"fnirs_{mod}_epochs.pt")).numpy()
        print(f"\n>>> 正在构建 {mod.upper()} 的四个 16x16 空间相关平面与直接耦合矩阵...")
        topo_dict = build_modality_planes(eeg_all, fnirs_all, eeg_coords, fnirs_coords, spatial_affinity)
        save_path = os.path.join(processed_dir, f"topo_{mod}.pt")
        torch.save(topo_dict, save_path)
        print(f">>> 已保存至 {save_path} | C_EF: {topo_dict['topo_ef'].shape}, Corr_EF: {topo_dict['corr_ef'].shape}")

if __name__ == "__main__":
    main()