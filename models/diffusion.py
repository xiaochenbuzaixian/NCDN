import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCDMDiffusion(nn.Module):
    """
    标准条件扩散模型 (Standard Conditional Diffusion Model):
    直接对目标 fNIRS 信号 f_0 执行前向加噪与逆向 DDIM 去噪，无独立先验分支与 0.06 缩放魔改
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

    def predict_eps_and_x0(self, e_0, f_t, t, topo_ef, topo_fe, corr_ef=None):
        """
        采用 v-prediction 参数化 (Salimans & Ho, ICLR 2022):
        在低信噪比 (t->T) 时等价于预测噪声 eps，在高信噪比 (t->0) 时等价于预测信号 x_0，
        彻底解决纯 eps 预测在 t->T 除零放大基线漂移、或纯 x0 预测缺乏高频去噪能力的问题
        """
        v_pred = self.unet(e_0, f_t, t, topo_ef, topo_fe, corr_ef)
        ab = self.alpha_bar[t].view(-1, 1, 1).clamp(min=1e-6, max=1.0 - 1e-6)
        sqrt_ab = torch.sqrt(ab)
        sqrt_one_minus_ab = torch.sqrt(1.0 - ab)

        eps_pred = sqrt_ab * v_pred + sqrt_one_minus_ab * f_t
        x0_pred = sqrt_ab * f_t - sqrt_one_minus_ab * v_pred
        return eps_pred, x0_pred, v_pred

    def compute_loss(self, e_0, f_0, topo_ef, topo_fe, corr_ef=None):
        B = e_0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=e_0.device).long()
        noise = torch.randn_like(f_0)
        f_t = self.q_sample(f_0, t, noise)

        ab = self.alpha_bar[t].view(-1, 1, 1).clamp(min=1e-6, max=1.0 - 1e-6)
        v_target = torch.sqrt(ab) * noise - torch.sqrt(1.0 - ab) * f_0

        eps_pred, x0_pred, v_pred = self.predict_eps_and_x0(e_0, f_t, t, topo_ef, topo_fe, corr_ef)

        # 1. 标准扩散噪声匹配损失 (等价于带 SNR 平衡的 ||eps - eps_theta||^2)
        noise_mse = F.mse_loss(eps_pred, noise)
        v_mse = F.mse_loss(v_pred, v_target)
        x0_mse = F.mse_loss(x0_pred, f_0)

        # 2. 试次级对比相关性约束 (减去 Batch 平均模板后计算余弦相似度，强迫模型保留每个试次的 LMI/RMI 差异！)
        if B > 2:
            x0_diff = x0_pred - x0_pred.mean(dim=0, keepdim=True)
            f0_diff = f_0 - f_0.mean(dim=0, keepdim=True)
            trial_spec_cos = F.cosine_similarity(x0_diff.flatten(1), f0_diff.flatten(1), dim=-1, eps=1e-6).mean()
        else:
            trial_spec_cos = torch.tensor(0.0, device=e_0.device)

        total_loss = v_mse + 0.5 * x0_mse + 0.2 * (1.0 - trial_spec_cos)
        return total_loss, {
            "total": total_loss.item(),
            "noise_mse": noise_mse.item(),
            "prior_mse": x0_mse.item(),
            "x0_mse": x0_mse.item()
        }

    @torch.no_grad()
    def sample(self, e_0, topo_ef, topo_fe, steps=50, corr_ef=None, eta=0.0):
        """标准 DDIM 逆向去噪采样 (从纯高斯噪声 f_T ~ N(0,I) 完整采样生成 f_0，无 0.06 截断)"""
        B, device = e_0.size(0), e_0.device
        seq = torch.linspace(0, self.timesteps - 1, steps + 1, dtype=torch.long, device=device).flip(0)
        ft = torch.randn(B, 36, 256, device=device)

        for i in range(len(seq) - 1):
            t_cur, t_next = seq[i], seq[i + 1]
            tt = torch.full((B,), t_cur, device=device, dtype=torch.long)

            eps_pred, x0_pred, _ = self.predict_eps_and_x0(e_0, ft, tt, topo_ef, topo_fe, corr_ef)
            x0_pred = x0_pred.clamp(-4.0, 4.0)

            ab_next = self.alpha_bar[t_next].clamp(min=1e-8)
            ft = torch.sqrt(ab_next) * x0_pred + torch.sqrt(torch.clamp(1.0 - ab_next, min=0.0)) * eps_pred

        # 仅将任务前 [-5s, 0s] (前50点) 静息基线归零，完整保留任务期幅值与空间通道分布
        ft = ft - ft[:, :, :50].mean(dim=-1, keepdim=True)
        return ft