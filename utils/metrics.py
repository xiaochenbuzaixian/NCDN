import numpy as np
import torch

def compute_generation_metrics(real_fnirs, syn_fnirs):
    """
    计算合成 fNIRS 与真实 fNIRS 的 RMSE 与平均皮尔逊相关系数 (PCC)
    输入形状: [N, 36, 256]
    """
    if isinstance(real_fnirs, torch.Tensor):
        real_fnirs = real_fnirs.detach().cpu().numpy()
    if isinstance(syn_fnirs, torch.Tensor):
        syn_fnirs = syn_fnirs.detach().cpu().numpy()
    rmse = np.sqrt(np.mean((real_fnirs - syn_fnirs) ** 2))
    # 计算样本平均血流动力学响应曲线的趋势相关性
    real_curve = np.mean(real_fnirs, axis=(0, 1))
    syn_curve = np.mean(syn_fnirs, axis=(0, 1))
    pcc = np.corrcoef(real_curve, syn_curve)[0, 1]
    return {"RMSE": float(rmse), "Curve_PCC": float(pcc)}