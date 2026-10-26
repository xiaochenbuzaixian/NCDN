import torch
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 启动冒烟测试 (SCDM U-Net + SCG + MTR) | 设备: {device}")
    B = 2
    e_hrf = torch.randn(B, 30, 256, device=device)
    f_0 = torch.randn(B, 36, 256, device=device)
    topo_ef = torch.randn(30, 16, 16, device=device)
    topo_fe = torch.randn(36, 16, 16, device=device)

    unet = SCDMUnet(time_dim=128, base_channels=64).to(device)
    diffusion = SCDMDiffusion(unet, timesteps=200, schedule="cosine").to(device)

    loss, l_dict = diffusion.compute_loss(e_hrf, f_0, topo_ef, topo_fe)
    loss.backward()
    print(f"  [通过] 前向与反向传播正常 | Noise MSE: {l_dict['noise_mse']:.4f} | Prior MSE: {l_dict['prior_mse']:.4f}")

    syn_f = diffusion.sample(e_hrf, topo_ef, topo_fe, steps=20)
    print(f"  [通过] DDIM 确定性采样正常 | 合成 fNIRS 形状: {syn_f.shape}")
    print("\n✅ 冒烟测试全部通过！")

if __name__ == "__main__":
    main()