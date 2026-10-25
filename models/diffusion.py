import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCDMDiffusion(nn.Module):
    def __init__(self, unet, timesteps=200, beta_start=1e-4, beta_end=0.02, schedule="cosine"):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps

        if schedule == "cosine":
            betas = self._cosine_schedule(timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, timesteps)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    @staticmethod
    def _cosine_schedule(T, s=0.008):
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def compute_loss(self, e_0, f_0, topo_ef, topo_fe, topo_ee=None, topo_ff=None):
        B = e_0.size(0)
        device = e_0.device

        # 1. 拓扑瓶颈先验预测
        f_prior = self.unet.forward_prior(e_0, topo_ef, topo_fe, topo_ee, topo_ff)

        prior_mse = F.mse_loss(f_prior, f_0)

        # 同时约束单试次全局相关性与逐通道时序相关性
        p_flat = f_prior.reshape(B, -1)
        t_flat = f_0.reshape(B, -1)
        p_norm = p_flat - p_flat.mean(dim=-1, keepdim=True)
        t_norm = t_flat - t_flat.mean(dim=-1, keepdim=True)
        sample_pcc = F.cosine_similarity(p_norm, t_norm, dim=-1, eps=1e-6).mean()

        ch_p_norm = f_prior - f_prior.mean(dim=-1, keepdim=True)
        ch_t_norm = f_0 - f_0.mean(dim=-1, keepdim=True)
        ch_pcc = F.cosine_similarity(ch_p_norm, ch_t_norm, dim=-1, eps=1e-6).mean()
        pcc_loss = 1.0 - (0.6 * sample_pcc + 0.4 * ch_pcc)

        curve_mse = F.mse_loss(f_prior.mean(dim=0), f_0.mean(dim=0))
        diff_loss = F.mse_loss(f_prior[:, :, 1:] - f_prior[:, :, :-1], f_0[:, :, 1:] - f_0[:, :, :-1])

        # 2. 对残差 r_0 = f_0 - f_prior.detach() 进行扩散去噪训练
        r_0 = f_0 - f_prior.detach()
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        noise_r = torch.randn_like(r_0)
        r_t = self.q_sample(r_0, t, noise_r)

        noise_pred = self.unet(e_0, r_t, t, topo_ef, topo_fe, topo_ee, topo_ff)
        noise_mse = F.mse_loss(noise_pred, noise_r)

        total_loss = noise_mse + 1.0 * prior_mse + 0.8 * pcc_loss + 2.0 * curve_mse + 5.0 * diff_loss
        loss_dict = {
            "total": total_loss.item(),
            "noise_mse": noise_mse.item(),
            "x0_mse": prior_mse.item(),
            "pcc_loss": (1.0 - sample_pcc).item()
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def sample(self, e_0, topo_ef, topo_fe, topo_ee=None, topo_ff=None, steps=50, eta=0.0):
        B = e_0.size(0)
        device = e_0.device

        f_prior = self.unet.forward_prior(e_0, topo_ef, topo_fe, topo_ee, topo_ff)

        steps = min(steps, self.timesteps)
        seq = torch.linspace(0, self.timesteps - 1, steps + 1, dtype=torch.long, device=device).flip(0)

        rt = torch.randn(B, 36, 256, device=device)
        for i in range(len(seq) - 1):
            t_cur = seq[i]
            t_next = seq[i + 1]
            tt = torch.full((B,), t_cur, device=device, dtype=torch.long)

            eps = self.unet(e_0, rt, tt, topo_ef, topo_fe, topo_ee, topo_ff)
            ab_cur = self.alpha_bar[t_cur].clamp(min=1e-8)
            ab_next = self.alpha_bar[t_next].clamp(min=1e-8)

            r0_pred = ((rt - torch.sqrt(1.0 - ab_cur) * eps) / torch.sqrt(ab_cur)).clamp(-2.5, 2.5)
            dir_rt = torch.sqrt(torch.clamp(1.0 - ab_next, min=0.0)) * eps
            rt = torch.sqrt(ab_next) * r0_pred + dir_rt

        # 低幅值残差注入，避免未见样本上的方差膨胀
        f_syn = f_prior + 0.05 * rt
        pad = 4
        f_syn = F.avg_pool1d(F.pad(f_syn, (pad, pad), mode='replicate'), kernel_size=9, stride=1)
        f_syn = f_syn - f_syn[:, :, :50].mean(dim=-1, keepdim=True)
        return f_syn