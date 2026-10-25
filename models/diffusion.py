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
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def compute_loss(self, e_0, f_0, topo_ef, topo_fe):
        B = e_0.shape[0]
        device = e_0.device
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        noise_f = torch.randn_like(f_0)
        noise_e = torch.randn_like(e_0)

        f_t = self.q_sample(f_0, t, noise_f)
        e_t = self.q_sample(e_0, t, noise_e)

        # 直接预测目标血氧信号 f_0 (避免 98% 高频白噪声捷径)
        pred_f0 = self.unet(e_t, f_t, t, topo_ef, topo_fe)

        # 1. 点对点幅值重建损失 (消除基线漂移)
        mse_loss = F.mse_loss(pred_f0, f_0)

        # 2. 时序波形相关性损失 (最大化皮尔逊相关系数 PCC)
        pred_norm = pred_f0 - pred_f0.mean(dim=-1, keepdim=True)
        targ_norm = f_0 - f_0.mean(dim=-1, keepdim=True)
        cos_sim = F.cosine_similarity(pred_norm, targ_norm, dim=-1, eps=1e-6).mean()
        pcc_loss = 1.0 - cos_sim

        # 3. 一阶导数平滑损失 (抑制高频毛刺，严格贴合低频血流动力学斜率)
        diff_pred = pred_f0[:, :, 1:] - pred_f0[:, :, :-1]
        diff_targ = f_0[:, :, 1:] - f_0[:, :, :-1]
        smooth_loss = F.mse_loss(diff_pred, diff_targ)

        return mse_loss + 0.5 * pcc_loss + 5.0 * smooth_loss

    @torch.no_grad()
    def sample(self, e_0, topo_ef, topo_fe, ddim_steps=50):
        """
        采用 DDIM 确定性逆向采样:
        1. 速度提升 20 倍 (50步替代1000步)
        2. 消除逐步随机白噪声叠加导致的高频锯齿
        """
        B = e_0.shape[0]
        device = e_0.device
        f_t = torch.randn((B, 36, 256), device=device)

        step_indices = torch.linspace(0, self.timesteps - 1, ddim_steps, device=device).long()
        step_indices = torch.flip(step_indices, dims=[0])

        for idx in range(len(step_indices)):
            i = step_indices[idx]
            t = torch.full((B,), i, device=device, dtype=torch.long)

            # 推理阶段使用干净 EEG 或对应时间步微弱加噪引导
            e_t = self.q_sample(e_0, t)
            pred_f0 = self.unet(e_t, f_t, t, topo_ef, topo_fe)
            pred_f0 = torch.clamp(pred_f0, -4.0, 4.0)

            if idx == len(step_indices) - 1:
                f_t = pred_f0
            else:
                i_prev = step_indices[idx + 1]
                alpha_bar_t = self.alphas_cumprod[i]
                alpha_bar_prev = self.alphas_cumprod[i_prev]

                # 从 pred_f0 反推等效噪声并执行 DDIM 确定性步进
                pred_eps = (f_t - torch.sqrt(alpha_bar_t) * pred_f0) / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)
                f_t = torch.sqrt(alpha_bar_prev) * pred_f0 + torch.sqrt(1.0 - alpha_bar_prev) * pred_eps

        # 生理频带平滑滤波 (滤除残余高频毛刺，严格对齐 0.01-0.1Hz 血流平滑度)
        kernel_size = 9
        pad = kernel_size // 2
        f_smooth = F.avg_pool1d(F.pad(f_t, (pad, pad), mode='replicate'), kernel_size=kernel_size, stride=1)
        return f_smooth