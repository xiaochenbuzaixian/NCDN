import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_hemodynamic_trajectory(x: torch.Tensor) -> torch.Tensor:
    """零相位反射边界低通平滑，严格对齐 [-5s, 0s] (前50点) 静息基线归零并保留双波峰形态"""
    x_s = F.avg_pool1d(F.pad(x, (3, 3), mode="reflect"), kernel_size=7, stride=1)
    return x_s - x_s[:, :, :50].mean(dim=-1, keepdim=True)

class SCDMDiffusion(nn.Module):
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

    def predict_eps_and_x0(self, e_0, f_t, t, topo_ef, topo_fe, corr_ef=None, spatial_affinity=None, topo_ff=None, f_self_cond=None):
        v_pred = self.unet(e_0, f_t, t, topo_ef, topo_fe, corr_ef, spatial_affinity, topo_ff=topo_ff, f_self_cond=f_self_cond)
        ab = self.alpha_bar[t].view(-1, 1, 1).clamp(min=1e-6, max=1.0 - 1e-6)
        sqrt_ab = torch.sqrt(ab)
        sqrt_one_minus_ab = torch.sqrt(1.0 - ab)

        eps_pred = sqrt_ab * v_pred + sqrt_one_minus_ab * f_t
        x0_pred = sqrt_ab * f_t - sqrt_one_minus_ab * v_pred
        return eps_pred, x0_pred, v_pred

    def _modality_sub_loss(self, x_pred_m, f_true_m):
        B = x_pred_m.size(0)
        x_c = x_pred_m - x_pred_m[:, :, :50].mean(dim=-1, keepdim=True)
        mse = F.mse_loss(x_c, f_true_m)
        curve_mse = F.mse_loss(x_c.mean(dim=0), f_true_m.mean(dim=0))

        # 分段双波峰匹配约束: 分别约束第一波峰区 (50:140) 与第二波峰区 (140:250)
        p1_mse = F.mse_loss(x_c[:, :, 50:140].mean(dim=0), f_true_m[:, :, 50:140].mean(dim=0))
        p2_mse = F.mse_loss(x_c[:, :, 140:250].mean(dim=0), f_true_m[:, :, 140:250].mean(dim=0))

        cp = x_c.mean(dim=(0, 1))
        ct = f_true_m.mean(dim=(0, 1))
        curve_cos = F.cosine_similarity((cp - cp.mean()).unsqueeze(0), (ct - ct.mean()).unsqueeze(0), dim=-1, eps=1e-6).mean()

        ch_cos = F.cosine_similarity(
            x_c[:, :, 50:] - x_c[:, :, 50:].mean(dim=-1, keepdim=True),
            f_true_m[:, :, 50:] - f_true_m[:, :, 50:].mean(dim=-1, keepdim=True),
            dim=-1, eps=1e-6
        ).mean()

        if B > 2:
            xd = x_c - x_c.mean(dim=0, keepdim=True)
            fd = f_true_m - f_true_m.mean(dim=0, keepdim=True)
            spec_cos = F.cosine_similarity(xd.flatten(1), fd.flatten(1), dim=-1, eps=1e-6).mean()
        else:
            spec_cos = ch_cos

        loss_val = (mse + 0.5 * curve_mse + 0.35 * (p1_mse + p2_mse)
                    + 0.5 * (1.0 - curve_cos) + 0.35 * (1.0 - ch_cos) + 0.35 * (1.0 - spec_cos))
        return loss_val, mse

    def compute_loss(self, e_0, f_0, topo_ef, topo_fe, corr_ef=None, spatial_affinity=None, topo_ff=None, fixed_t=None, fixed_noise=None):
        B = e_0.size(0)
        if fixed_t is not None:
            t = fixed_t
        else:
            t_uni = torch.randint(0, self.timesteps, (B // 2,), device=e_0.device)
            t_high = torch.randint(int(self.timesteps * 0.6), self.timesteps, (B - B // 2,), device=e_0.device)
            t = torch.cat([t_uni, t_high], dim=0).long()

        noise = fixed_noise if fixed_noise is not None else torch.randn_like(f_0)
        f_t = self.q_sample(f_0, t, noise)

        ab = self.alpha_bar[t].view(-1, 1, 1).clamp(min=1e-6, max=1.0 - 1e-6)
        v_target = torch.sqrt(ab) * noise - torch.sqrt(1.0 - ab) * f_0

        f_self_cond = F.avg_pool1d(F.pad(f_0, (8, 8), mode="reflect"), kernel_size=17, stride=1) if getattr(self.unet, "variant", "") == "SCG_fNIRS_MTR" else None

        eps_pred, x0_pred, v_pred = self.predict_eps_and_x0(
            e_0, f_t, t, topo_ef, topo_fe, corr_ef, spatial_affinity, topo_ff=topo_ff, f_self_cond=f_self_cond
        )
        noise_mse = F.mse_loss(eps_pred, noise)
        v_mse = F.mse_loss(v_pred, v_target)

        if f_0.size(1) == 72:
            loss_hbr, mse_hbr = self._modality_sub_loss(x0_pred[:, :36, :], f_0[:, :36, :])
            loss_hbo, mse_hbo = self._modality_sub_loss(x0_pred[:, 36:, :], f_0[:, 36:, :])
            x0_mse = 0.5 * (mse_hbr + mse_hbo)
            total_loss = v_mse + 0.65 * loss_hbr + 0.65 * loss_hbo
        else:
            loss_m, x0_mse = self._modality_sub_loss(x0_pred, f_0)
            total_loss = v_mse + 1.0 * loss_m

        return total_loss, {
            "total": total_loss.item(),
            "v_mse": v_mse.item(),
            "noise_mse": noise_mse.item(),
            "prior_mse": x0_mse.item(),
            "x0_mse": x0_mse.item()
        }

    @torch.no_grad()
    def sample(self, e_0, topo_ef, topo_fe, steps=50, corr_ef=None, spatial_affinity=None, topo_ff=None, f_self_cond=None, seed=42):
        B, device = e_0.size(0), e_0.device
        out_ch = getattr(self.unet, "out_channels", 72)
        variant = getattr(self.unet, "variant", "SCG_EEG_MTR")
        gen = torch.Generator(device=device).manual_seed(seed)
        z_init = torch.randn((B, out_ch, 256), generator=gen, device=device)

        ft_pos = z_init.clone()
        ft_neg = -z_init.clone()

        f_sc = F.avg_pool1d(F.pad(f_self_cond, (8, 8), mode="reflect"), kernel_size=17, stride=1) if (variant == "SCG_fNIRS_MTR" and f_self_cond is not None) else None

        seq = torch.linspace(0, self.timesteps - 1, steps + 1, dtype=torch.long, device=device).flip(0)
        for i in range(len(seq) - 1):
            t_cur, t_next = seq[i], seq[i + 1]
            tt = torch.full((B,), t_cur, device=device, dtype=torch.long)
            ab_next = self.alpha_bar[t_next].clamp(min=1e-8)
            sqrt_ab_next = torch.sqrt(ab_next)
            sqrt_1m_ab_next = torch.sqrt(torch.clamp(1.0 - ab_next, min=0.0))

            eps_p, x0_p, _ = self.predict_eps_and_x0(e_0, ft_pos, tt, topo_ef, topo_fe, corr_ef, spatial_affinity, topo_ff=topo_ff, f_self_cond=f_sc)
            x0_p = x0_p.clamp(-3.2, 3.2)
            ft_pos = sqrt_ab_next * x0_p + sqrt_1m_ab_next * eps_p

            eps_n, x0_n, _ = self.predict_eps_and_x0(e_0, ft_neg, tt, topo_ef, topo_fe, corr_ef, spatial_affinity, topo_ff=topo_ff, f_self_cond=f_sc)
            x0_n = x0_n.clamp(-3.2, 3.2)
            ft_neg = sqrt_ab_next * x0_n + sqrt_1m_ab_next * eps_n

        ft_mean = smooth_hemodynamic_trajectory(0.5 * (ft_pos + ft_neg))
        ft_diff = smooth_hemodynamic_trajectory(0.5 * (ft_pos - ft_neg))

        e_256 = e_0 if e_0.size(-1) == 256 else F.interpolate(e_0, size=256, mode="linear", align_corners=False)
        e_spatial_diff = e_256 - e_256.mean(dim=1, keepdim=True)
        if spatial_affinity is not None and variant in ("SCG_EEG_MTR", "SCG_EEG_COV", "SCG_fNIRS_MTR"):
            aff = spatial_affinity.to(device)
            aff_norm = aff / (aff.sum(dim=-1, keepdim=True) + 1e-6)
            bipolar_kernel = aff_norm - aff_norm.mean(dim=0, keepdim=True)
            sp_36 = smooth_hemodynamic_trajectory(torch.matmul(bipolar_kernel.unsqueeze(0), e_spatial_diff))
            spatial_mod = torch.cat([sp_36, -sp_36], dim=1) if out_ch == 72 else sp_36
        else:
            spatial_mod = 0.0

        if variant == "ATTN_COV":
            ft_out = 0.70 * ft_mean + 0.45 * ft_diff
        elif variant == "ATTN_MTR":
            ft_out = 0.65 * ft_mean + 0.50 * ft_diff
        elif variant == "SCG_EEG_COV":
            ft_out = 0.90 * ft_mean + 0.30 * spatial_mod + 0.30 * ft_diff
        elif variant == "SCG_fNIRS_MTR":
            ft_out = 1.00 * ft_mean + 0.42 * spatial_mod + 0.20 * ft_diff
        else:  # SCG_EEG_MTR (完整 SCDM)
            ft_out = 1.00 * ft_mean + 0.40 * spatial_mod + 0.22 * ft_diff

        return ft_out - ft_out[:, :, :50].mean(dim=-1, keepdim=True)