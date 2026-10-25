import torch
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 启动冒烟测试 | 当前计算设备: {device}")

    unet = SCDMUnet(time_dim=64).to(device)
    diffusion = SCDMDiffusion(unet, timesteps=1000).to(device)

    # 模拟输入张量
    B = 4
    e_0 = torch.randn(B, 30, 4000, device=device)
    f_0 = torch.randn(B, 36, 256, device=device)
    topo_ef = torch.randn(30, 16, 16, device=device)
    topo_fe = torch.randn(36, 16, 16, device=device)

    # 1. 验证训练 Loss 与梯度回传
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss = diffusion.compute_loss(e_0, f_0, topo_ef, topo_fe)
    loss.backward()
    optimizer.step()
    print(f"  [通过] 前向与反向传播正常 | 初始 MSE Loss: {loss.item():.4f}")

    # 2. 验证逆向去噪生成 (临时设为10步快速验证)
    diffusion.timesteps = 10
    syn_f = diffusion.sample(e_0[:2], topo_ef, topo_fe)
    assert syn_f.shape == (2, 36, 256), f"尺寸错误: {syn_f.shape}"
    print(f"  [通过] 逆向采样生成正常 | 合成 fNIRS 形状: {syn_f.shape}")
    print("\n✅ 冒烟测试全部通过！可以开始运行预处理与正式训练。")

if __name__ == "__main__":
    main()