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
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # 后验分布 q(x_{t-1} | x_t, x_0) 的解析系数，彻底杜绝采样数值爆炸
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def predict_start_from_noise(self, f_t, t, pred_noise):
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return (f_t - sqrt_one_minus_alpha_bar * pred_noise) / (sqrt_alpha_bar + 1e-8)

    def compute_loss(self, e_0, f_0, topo_ef, topo_fe):
        B = e_0.shape[0]
        device = e_0.device
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        noise_f = torch.randn_like(f_0)
        noise_e = torch.randn_like(e_0)

        f_t = self.q_sample(f_0, t, noise_f)
        # 对条件 EEG 施加温和噪声，保留跨模态引导信息
        e_t = self.q_sample(e_0, t // 2, noise_e)

        pred_noise = self.unet(e_t, f_t, t, topo_ef, topo_fe)
        noise_loss = F.mse_loss(pred_noise, noise_f)

        # 辅助重建约束：引导小时间步下的信号波形趋势一致
        f_0_pred = torch.clamp(self.predict_start_from_noise(f_t, t, pred_noise), -5.0, 5.0)
        weight = (self.alphas_cumprod[t] ** 0.5).view(-1, 1, 1)
        recon_loss = F.l1_loss(f_0_pred * weight, f_0 * weight)

        return noise_loss + 0.1 * recon_loss

    @torch.no_grad()
    def sample(self, e_0, topo_ef, topo_fe):
        B = e_0.shape[0]
        device = e_0.device
        f_t = torch.randn((B, 36, 256), device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            e_t = self.q_sample(e_0, t // 2)

            pred_noise = self.unet(e_t, f_t, t, topo_ef, topo_fe)

            # 1. 重建 f_0 并进行生理幅值截断保护 (防止 1000 步累积爆炸)
            f_0_pred = self.predict_start_from_noise(f_t, t, pred_noise)
            f_0_pred = torch.clamp(f_0_pred, -4.0, 4.0)

            # 2. 使用稳定的后验均值公式计算 f_{t-1}
            coef1 = self.posterior_mean_coef1[i]
            coef2 = self.posterior_mean_coef2[i]
            posterior_mean = coef1 * f_0_pred + coef2 * f_t

            if i > 0:
                var = self.posterior_variance[i]
                z = torch.randn_like(f_t)
                f_t = posterior_mean + torch.sqrt(var) * z
            else:
                f_t = posterior_mean

        return f_t