import torch
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 启动冒烟测试 (6-Block 标准 SCDM: 32->16->8->4->8->16->32) | 设备: {device}")
    B = 2
    e_hrf = torch.randn(B, 30, 256, device=device)
    f_0 = torch.randn(B, 36, 256, device=device)
    topo_ef = torch.randn(30, 16, 16, device=device)
    topo_fe = torch.randn(36, 16, 16, device=device)
    corr_ef = torch.rand(30, 36, device=device)

    unet = SCDMUnet(time_dim=64, base_channels=32, n_blocks=6).to(device)
    diffusion = SCDMDiffusion(unet, timesteps=200, schedule="cosine").to(device)

    loss, l_dict = diffusion.compute_loss(e_hrf, f_0, topo_ef, topo_fe, corr_ef=corr_ef)
    loss.backward()
    print(f"  [通过] 6-Block U-Net 前向与反向传播正常 | Noise MSE: {l_dict['noise_mse']:.4f} | x0 MSE: {l_dict['x0_mse']:.4f}")

    syn_f = diffusion.sample(e_hrf, topo_ef, topo_fe, steps=20, corr_ef=corr_ef)
    print(f"  [通过] 标准 DDIM 采样正常 (无 0.06 残差缩放) | 合成 fNIRS 形状: {syn_f.shape}")
    print("\n✅ 冒烟测试通过！")

if __name__ == "__main__":
    main()