import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_hemodynamic_trajectory(x: torch.Tensor) -> torch.Tensor:
    """反射边界延拓零相位低通平滑 (消除 DDIM 采样高频毛刺，严格对齐 [-5s, 0s] 零基线)"""
    x_s = F.avg_pool1d(F.pad(x, (6, 6), mode="reflect"), kernel_size=13, stride=1)
    x_s = F.avg_pool1d(F.pad(x_s, (4, 4), mode="reflect"), kernel_size=9, stride=1)
    return x_s - x_s[:, :, :50].mean(dim=-1, keepdim=True)

class SCDMDiffusion(nn.Module):
    """
    标准 6-Block 条件扩散模型:
    解决 "训练 x0_MSE=0.15 但采样 Val_MSE=1.23" 的训练-采样不一致问题
    """
    def __init__(self, unet, timesteps=200, beta_start=1e-4, beta_end=0.02, schedule="cosine"):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps

        if schedule == "cosine":
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps) / timesteps
            alphas_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = torch.clamp(1 - (alphas_bar[1:] / alphas_bar[:-1]), 0.0001, 0.9999)
        else:
            betas = torch.linspace(beta_start, beta_end, timesteps)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def predict_eps_and_x0(self, e_0, f_t, t, topo_ef, topo_fe, corr_ef=None, spatial_affinity=None):
        v_pred = self.unet(e_0, f_t, t, topo_ef, topo_fe, corr_ef, spatial_affinity)
        ab = self.alpha_bar[t].view(-1, 1, 1).clamp(min=1e-6, max=1.0 - 1e-6)
        sqrt_ab = torch.sqrt(ab)
        sqrt_one_minus_ab = torch.sqrt(1.0 - ab)

        eps_pred = sqrt_ab * v_pred + sqrt_one_minus_ab * f_t
        x0_pred = sqrt_ab * f_t - sqrt_one_minus_ab * v_pred
        return eps_pred, x0_pred, v_pred

    def compute_loss(self, e_0, f_0, topo_ef, topo_fe, corr_ef=None, spatial_affinity=None, fixed_t=None, fixed_noise=None):
        B = e_0.size(0)
        if fixed_t is not None:
            t = fixed_t
        else:
            # 50% 均匀采样全时间步，50% 重点采样高噪声步 t in [120, 199] (强迫模型在纯噪声起点依赖 EEG 条件恢复信号)
            t_uni = torch.randint(0, self.timesteps, (B // 2,), device=e_0.device)
            t_high = torch.randint(int(self.timesteps * 0.6), self.timesteps, (B - B // 2,), device=e_0.device)
            t = torch.cat([t_uni, t_high], dim=0).long()

        noise = fixed_noise if fixed_noise is not None else torch.randn_like(f_0)
        f_t = self.q_sample(f_0, t, noise)

        ab = self.alpha_bar[t].view(-1, 1, 1).clamp(min=1e-6, max=1.0 - 1e-6)
        v_target = torch.sqrt(ab) * noise - torch.sqrt(1.0 - ab) * f_0

        eps_pred, x0_pred, v_pred = self.predict_eps_and_x0(e_0, f_t, t, topo_ef, topo_fe, corr_ef, spatial_affinity)

        noise_mse = F.mse_loss(eps_pred, noise)
        v_mse = F.mse_loss(v_pred, v_target)
        x0_mse = F.mse_loss(x0_pred, f_0)
        curve_mse = F.mse_loss(x0_pred.mean(dim=0), f_0.mean(dim=0))

        # 逐试次相关性与试次间特异性约束 (强迫保留单试次左右脑偏侧化差异)
        x_flat = x0_pred.reshape(B, -1)
        f_flat = f_0.reshape(B, -1)
        sample_cos = F.cosine_similarity(
            x_flat - x_flat.mean(dim=-1, keepdim=True),
            f_flat - f_flat.mean(dim=-1, keepdim=True),
            dim=-1, eps=1e-6
        ).mean()

        if B > 2:
            x0_diff = x0_pred - x0_pred.mean(dim=0, keepdim=True)
            f0_diff = f_0 - f_0.mean(dim=0, keepdim=True)
            trial_spec_cos = F.cosine_similarity(x0_diff.flatten(1), f0_diff.flatten(1), dim=-1, eps=1e-6).mean()
        else:
            trial_spec_cos = sample_cos

        total_loss = v_mse + 0.8 * x0_mse + 0.5 * curve_mse + 0.25 * (1.0 - sample_cos) + 0.25 * (1.0 - trial_spec_cos)
        return total_loss, {
            "total": total_loss.item(),
            "v_mse": v_mse.item(),
            "noise_mse": noise_mse.item(),
            "prior_mse": x0_mse.item(),
            "x0_mse": x0_mse.item()
        }

    @torch.no_grad()
    def sample(self, e_0, topo_ef, topo_fe, steps=50, corr_ef=None, spatial_affinity=None, seed=42):
        """固定种子的确定性 DDIM 逆向去噪采样"""
        B, device = e_0.size(0), e_0.device
        gen = torch.Generator(device=device).manual_seed(seed)
        ft = torch.randn((B, 36, 256), generator=gen, device=device)

        seq = torch.linspace(0, self.timesteps - 1, steps + 1, dtype=torch.long, device=device).flip(0)
        for i in range(len(seq) - 1):
            t_cur, t_next = seq[i], seq[i + 1]
            tt = torch.full((B,), t_cur, device=device, dtype=torch.long)

            eps_pred, x0_pred, _ = self.predict_eps_and_x0(e_0, ft, tt, topo_ef, topo_fe, corr_ef, spatial_affinity)
            x0_pred = x0_pred.clamp(-3.0, 3.0)

            ab_next = self.alpha_bar[t_next].clamp(min=1e-8)
            ft = torch.sqrt(ab_next) * x0_pred + torch.sqrt(torch.clamp(1.0 - ab_next, min=0.0)) * eps_pred

        return smooth_hemodynamic_trajectory(ft)