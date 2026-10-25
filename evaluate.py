import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.shin_dataset import get_dataloaders
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion
from scipy.stats import pearsonr


def evaluate_checkpoint(epoch=60):
    with open("configs/shin_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    modality = cfg["data"]["target_modality"]
    processed_dir = cfg["data"]["processed_dir"]
    ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_ep{epoch}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    # 1. 加载拓扑与数据
    topo_dict = torch.load(os.path.join(processed_dir, f"topo_{modality.lower()}.pt"), map_location="cpu")
    topo_ef = topo_dict["topo_ef"].to(device)
    topo_fe = topo_dict["topo_fe"].to(device)

    _, val_loader = get_dataloaders(
        processed_dir=processed_dir,
        modality=modality,
        batch_size=cfg["data"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"]
    )

    # 2. 初始化模型并加载黄金权重
    unet = SCDMUnet(time_dim=cfg["model"]["time_dim"]).to(device)
    model = SCDMDiffusion(
        unet,
        timesteps=cfg["model"]["timesteps"],
        beta_start=cfg["model"]["beta_start"],
        beta_end=cfg["model"]["beta_end"]
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print(f">>> 开始对 {ckpt_path} 进行全量验证集评估...")

    all_real = []
    all_syn = []

    # 3. 遍历完整验证集进行推理
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="全量推理中"):
            e_0 = batch["eeg"].to(device)
            f_0 = batch["fnirs"].cpu().numpy()

            # 使用扩散模型逆向采样生成
            syn_f = model.sample(e_0, topo_ef, topo_fe).cpu().numpy()

            all_real.append(f_0)
            all_syn.append(syn_f)

    all_real = np.concatenate(all_real, axis=0)  # [N_val, 36, 256]
    all_syn = np.concatenate(all_syn, axis=0)  # [N_val, 36, 256]

    # 4. 计算全集严格指标
    global_rmse = np.sqrt(np.mean((all_real - all_syn) ** 2))

    real_mean_curve = np.mean(all_real, axis=(0, 1))
    syn_mean_curve = np.mean(all_syn, axis=(0, 1))
    global_pcc, _ = pearsonr(real_mean_curve, syn_mean_curve)

    print("\n" + "=" * 40)
    print(f"评估完成 | 验证集试次总数: {all_real.shape[0]}")
    print(f"全量 RMSE: {global_rmse:.4f}")
    print(f"全量 PCC:  {global_pcc:.4f}")
    print("=" * 40)

    # 5. 绘制并保存对比曲线
    plt.figure(figsize=(10, 4))
    time_axis = np.linspace(0, 25.6, 256)
    plt.plot(time_axis, real_mean_curve, label='Real fNIRS (Ground Truth)', color='blue', linewidth=2)
    plt.plot(time_axis, syn_mean_curve, label='Synthetic fNIRS (SCDM)', color='red', linestyle='--', linewidth=2)
    plt.title(f"Global Average fNIRS Response (Epoch {epoch}) | PCC: {global_pcc:.4f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Z-score)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(cfg["train"]["output_dir"], f"eval_curve_ep{epoch}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f">>> 对比曲线图已保存至: {plot_path}")


if __name__ == "__main__":
    # 优先评估 RMSE 最小的第 60 轮
    evaluate_checkpoint(epoch=60)

    # 也可取消注释，对比第 40 轮
    # evaluate_checkpoint(epoch=40)