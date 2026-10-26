import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCDMDiffusion(nn.Module):
    def __init__(self, unet, timesteps=200, beta_start=1e-4, beta_end=0.02, schedule="cosine"):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps

        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = torch.clamp(1 - (alphas_bar[1:] / alphas_bar[:-1]), 0.0001, 0.9999)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0))

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def compute_loss(self, e_0, f_0, topo_ef, topo_fe, *args):
        B = e_0.size(0)
        # 1. 物理先验与单试次相关性约束
        f_prior = self.unet.forward_prior(e_0, topo_ef, topo_fe)
        prior_mse = F.mse_loss(f_prior, f_0)
        curve_mse = F.mse_loss(f_prior.mean(dim=0), f_0.mean(dim=0))

        # 单试次皮尔逊相关性损失与时序一阶差分平滑损失
        p_flat = f_prior.reshape(B, -1)
        t_flat = f_0.reshape(B, -1)
        p_norm = p_flat - p_flat.mean(dim=-1, keepdim=True)
        t_norm = t_flat - t_flat.mean(dim=-1, keepdim=True)
        sample_pcc = F.cosine_similarity(p_norm, t_norm, dim=-1, eps=1e-6).mean()
        pcc_loss = 1.0 - sample_pcc

        diff_loss = F.mse_loss(f_prior[:, :, 1:] - f_prior[:, :, :-1], f_0[:, :, 1:] - f_0[:, :, :-1])

        # 2. 扩散残差去噪损失
        r_0 = f_0 - f_prior.detach()
        t = torch.randint(0, self.timesteps, (B,), device=e_0.device).long()
        noise = torch.randn_like(r_0)
        r_t = self.q_sample(r_0, t, noise)

        noise_pred = self.unet(e_0, r_t, t, topo_ef, topo_fe)
        noise_mse = F.mse_loss(noise_pred, noise)

        total_loss = noise_mse + 1.5 * prior_mse + 0.5 * pcc_loss + 2.0 * curve_mse + 3.0 * diff_loss
        return total_loss, {
            "total": total_loss.item(),
            "noise_mse": noise_mse.item(),
            "prior_mse": prior_mse.item(),
            "x0_mse": prior_mse.item()
        }

    @torch.no_grad()
    def sample(self, e_0, topo_ef, topo_fe, steps=50, *args):
        B, device = e_0.size(0), e_0.device
        f_prior = self.unet.forward_prior(e_0, topo_ef, topo_fe)
        seq = torch.linspace(0, self.timesteps - 1, steps + 1, dtype=torch.long, device=device).flip(0)
        rt = torch.randn(B, 36, 256, device=device)

        for i in range(len(seq) - 1):
            t_cur, t_next = seq[i], seq[i + 1]
            tt = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps = self.unet(e_0, rt, tt, topo_ef, topo_fe)
            ab_cur = self.alpha_bar[t_cur].clamp(min=1e-8)
            ab_next = self.alpha_bar[t_next].clamp(min=1e-8)

            r0_pred = ((rt - torch.sqrt(1.0 - ab_cur) * eps) / torch.sqrt(ab_cur)).clamp(-2.5, 2.5)
            rt = torch.sqrt(ab_next) * r0_pred + torch.sqrt(torch.clamp(1.0 - ab_next, min=0.0)) * eps

        f_syn = f_prior + 0.06 * rt
        f_syn = F.avg_pool1d(F.pad(f_syn, (4, 4), mode='replicate'), kernel_size=9, stride=1)
        # 关键修复：保持全时段零均值，与 preprocess_shin.py 中的真实信号严格一致！
        return f_syn - f_syn.mean(dim=-1, keepdim=True)