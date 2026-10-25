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

    def generate_structured_noise(self, x):
        white = torch.randn_like(x)
        pad = 12
        low = F.avg_pool1d(F.pad(white, (pad, pad), mode='reflect'), kernel_size=25, stride=1)
        low = F.avg_pool1d(F.pad(low, (pad, pad), mode='reflect'), kernel_size=25, stride=1)
        low = (low - low.mean(dim=-1, keepdim=True)) / (low.std(dim=-1, keepdim=True) + 1e-6)
        return 0.85 * low + 0.15 * white

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def compute_loss(self, e_0, f_0, topo_ef, topo_fe):
        B = e_0.shape[0]
        device = e_0.device
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        noise_f = self.generate_structured_noise(f_0)
        noise_e = torch.randn_like(e_0)

        f_t = self.q_sample(f_0, t, noise_f)
        # 50% 模态掩码：防止模型依赖 f_t 低频分量偷懒
        mask = (torch.rand(B, 1, 1, device=device) > 0.5).float()
        f_t = mask * f_t + (1.0 - mask) * noise_f

        e_t = self.q_sample(e_0, t // 10, noise_e)
        pred_f0 = self.unet(e_t, f_t, t, topo_ef, topo_fe)

        # 1. SNR 加权单试次点对点 MSE (降低大时间步纯噪声下的梯度震荡)
        snr_weight = (0.2 + 0.8 * self.alphas_cumprod[t]).view(-1, 1, 1)
        trial_mse = torch.mean(snr_weight * (pred_f0 - f_0) ** 2)

        # 2. 单试次 + 批次波形余弦相关性损失 (PCC Loss)
        p_norm = pred_f0 - pred_f0.mean(dim=-1, keepdim=True)
        t_norm = f_0 - f_0.mean(dim=-1, keepdim=True)
        trial_cos = F.cosine_similarity(p_norm, t_norm, dim=-1, eps=1e-6).mean()

        curve_pred = pred_f0.mean(dim=(0, 1))
        curve_targ = f_0.mean(dim=(0, 1))
        cp_norm = curve_pred - curve_pred.mean()
        ct_norm = curve_targ - curve_targ.mean()
        batch_cos = F.cosine_similarity(cp_norm.unsqueeze(0), ct_norm.unsqueeze(0), dim=-1).mean()
        pcc_loss = 1.0 - (0.4 * trial_cos + 0.6 * batch_cos)

        # 3. 批次幅值校准 + 一阶导数平滑损失 (对齐幅值与斜率)
        amp_loss = F.mse_loss(pred_f0.mean(dim=0), f_0.mean(dim=0))
        diff_pred = pred_f0[:, :, 1:] - pred_f0[:, :, :-1]
        diff_targ = f_0[:, :, 1:] - f_0[:, :, :-1]
        smooth_loss = F.mse_loss(diff_pred, diff_targ)

        total_loss = trial_mse + 1.0 * amp_loss + 0.3 * pcc_loss + 2.0 * smooth_loss

        loss_dict = {
            "total": total_loss.item(),
            "mse": trial_mse.item(),
            "amp": amp_loss.item(),
            "pcc": pcc_loss.item(),
            "smooth": smooth_loss.item()
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def sample(self, e_0, topo_ef, topo_fe, ddim_steps=20, noise_scale=0.25):
        """
        引入 noise_scale 控制采样方差膨胀：
        避免初始随机慢波淹没 EEG 诱发的真实幅值，将全集 RMSE 降至 < 1.0
        """
        B = e_0.shape[0]
        device = e_0.device
        f_t = noise_scale * self.generate_structured_noise(torch.zeros(B, 36, 256, device=device))

        step_indices = torch.linspace(0, self.timesteps - 1, ddim_steps, device=device).long()
        step_indices = torch.flip(step_indices, dims=[0])

        for idx in range(len(step_indices)):
            i = step_indices[idx]
            t = torch.full((B,), i, device=device, dtype=torch.long)

            e_t = self.q_sample(e_0, t // 10)
            pred_f0 = self.unet(e_t, f_t, t, topo_ef, topo_fe)
            pred_f0 = torch.clamp(pred_f0, -3.5, 3.5)

            if idx == len(step_indices) - 1:
                f_t = pred_f0
            else:
                i_prev = step_indices[idx + 1]
                alpha_bar_t = self.alphas_cumprod[i]
                alpha_bar_prev = self.alphas_cumprod[i_prev]

                pred_eps = (f_t - torch.sqrt(alpha_bar_t) * pred_f0) / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)
                f_t = torch.sqrt(alpha_bar_prev) * pred_f0 + noise_scale * torch.sqrt(1.0 - alpha_bar_prev) * pred_eps

        kernel_size = 9
        pad = kernel_size // 2
        f_smooth = F.avg_pool1d(F.pad(f_t, (pad, pad), mode='replicate'), kernel_size=kernel_size, stride=1)
        return f_smooth