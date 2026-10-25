import torch
import torch.nn as nn
import torch.nn.functional as F

class SCDMDiffusion(nn.Module):
    def __init__(self, unet, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        """前向加噪过程: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def compute_loss(self, e_0, f_0, topo_ef, topo_fe):
        """Algorithm 1: 计算预测噪声与真实噪声的 L2 损失"""
        B = e_0.shape[0]
        device = e_0.device
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        noise_f = torch.randn_like(f_0)
        noise_e = torch.randn_like(e_0)

        f_t = self.q_sample(f_0, t, noise_f)
        e_t = self.q_sample(e_0, t, noise_e)

        pred_noise = self.unet(e_t, f_t, t, topo_ef, topo_fe)
        loss = F.mse_loss(pred_noise, noise_f)
        return loss

    @torch.no_grad()
    def sample(self, e_0, topo_ef, topo_fe):
        """Algorithm 2: 给定原始 EEG e_0 逆向去噪合成 fNIRS \hat{f}_0"""
        B = e_0.shape[0]
        device = e_0.device
        f_t = torch.randn((B, 36, 256), device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            e_t = self.q_sample(e_0, t)

            pred_noise = self.unet(e_t, f_t, t, topo_ef, topo_fe)

            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            f_t = coef1 * (f_t - coef2 * pred_noise)

            if i > 0:
                z = torch.randn_like(f_t)
                f_t = f_t + torch.sqrt(beta_t) * z

        return f_t