import os
import torch
import numpy as np
import scipy.io as sio
import dcor
from tqdm import tqdm

def get_sensor_grid_coords(raw_dir="datasets/raw"):
    """
    读取 mnt.mat 中的 2D 头皮坐标 (x, y) 并离散化映射到 16x16 平面网格
    若读取失败则采用固定随机拓扑分布
    """
    grid_eeg, grid_fnirs = {}, {}
    try:
        e_mnt_path = os.path.join(raw_dir, "EEG", "subject 01", "with occular artifact", "mnt.mat")
        if not os.path.exists(e_mnt_path):
            e_mnt_path = os.path.join(raw_dir, "EEG", "subject 01", "mnt_artifact.mat")
        f_mnt_path = os.path.join(raw_dir, "fNIRS", "subject 01", "mnt.mat")

        e_mnt = sio.loadmat(e_mnt_path, struct_as_record=False, squeeze_me=True)['mnt']
        f_mnt = sio.loadmat(f_mnt_path, struct_as_record=False, squeeze_me=True)['mnt']

        ex, ey = e_mnt.x[:30], e_mnt.y[:30]
        fx, fy = f_mnt.x[:36], f_mnt.y[:36]

        all_x = np.concatenate([ex, fx])
        all_y = np.concatenate([ey, fy])
        min_x, max_x = all_x.min(), all_x.max()
        min_y, max_y = all_y.min(), all_y.max()

        for i in range(30):
            r = int(np.clip((ey[i] - min_y) / (max_y - min_y + 1e-6) * 15, 0, 15))
            c = int(np.clip((ex[i] - min_x) / (max_x - max_x + 1e-6) * 15, 0, 15))
            grid_eeg[i] = (r, c)
        for j in range(36):
            r = int(np.clip((fy[j] - min_y) / (max_y - min_y + 1e-6) * 15, 0, 15))
            c = int(np.clip((fx[j] - min_x) / (max_x - min_x + 1e-6) * 15, 0, 15))
            grid_fnirs[j] = (r, c)
    except Exception:
        np.random.seed(42)
        coords = [(r, c) for r in range(1, 15) for c in range(1, 15)]
        chosen = np.random.choice(len(coords), 66, replace=False)
        for i in range(30):
            grid_eeg[i] = coords[chosen[i]]
        for j in range(36):
            grid_fnirs[j] = coords[chosen[30 + j]]

    return grid_eeg, grid_fnirs

def build_topology_matrices(processed_dir="datasets/processed", modality="HbR"):
    eeg_path = os.path.join(processed_dir, "eeg_epochs.pt")
    fnirs_file = "fnirs_hbr_epochs.pt" if modality == "HbR" else "fnirs_hbo_epochs.pt"
    fnirs_path = os.path.join(processed_dir, fnirs_file)

    eeg = torch.load(eeg_path).numpy()      # [N, 30, 4000]
    fnirs = torch.load(fnirs_path).numpy()  # [N, 36, 256]

    print(f">>> 正在计算 EEG 与 {modality} 的通道间距离相关系数矩阵 (30x36)...")
    # 取样本均值代表稳态序列计算跨长度相关性
    e_avg = np.mean(eeg[:300], axis=0)      # [30, 4000]
    f_avg = np.mean(fnirs[:300], axis=0)    # [36, 256]

    C_ef = np.zeros((30, 36), dtype=np.float32)
    for i in tqdm(range(30)):
        for j in range(36):
            # 将 EEG 下采样对齐序列分布以加速 dcor 计算或分块计算
            e_sub = e_avg[i, ::15][:256]
            C_ef[i, j] = dcor.distance_correlation(e_sub, f_avg[j])

    C_fe = C_ef.T  # [36, 30]
    grid_eeg, grid_fnirs = get_sensor_grid_coords()

    # 投影到 16x16 平面: C_ef -> [30, 16, 16], C_fe -> [36, 16, 16]
    topo_ef = np.zeros((30, 16, 16), dtype=np.float32)
    for i in range(30):
        for j in range(36):
            r, c = grid_fnirs[j]
            topo_ef[i, r, c] = C_ef[i, j]

    topo_fe = np.zeros((36, 16, 16), dtype=np.float32)
    for j in range(36):
        for i in range(30):
            r, c = grid_eeg[i]
            topo_fe[j, r, c] = C_fe[j, i]

    save_path = os.path.join(processed_dir, f"topo_{modality.lower()}.pt")
    torch.save({
        "C_ef": torch.tensor(C_ef),
        "C_fe": torch.tensor(C_fe),
        "topo_ef": torch.tensor(topo_ef),
        "topo_fe": torch.tensor(topo_fe)
    }, save_path)
    print(f">>> 空间拓扑矩阵已保存至: {save_path}")

if __name__ == "__main__":
    build_topology_matrices(modality="HbR")
    build_topology_matrices(modality="HbO")