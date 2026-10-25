import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, 5, padding=2)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, cond_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(cond_emb)[..., None]
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return x + h

class PlanesEncoder(nn.Module):
    def __init__(self, cond_dim: int = 128):
        super().__init__()
        self.cross_enc = nn.Sequential(
            nn.Conv2d(30 + 36, 32, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.uni_enc = nn.Sequential(
            nn.Conv2d(30 + 36, 32, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, topo_ef, topo_fe, topo_ee=None, topo_ff=None, B=1):
        if topo_ef.dim() == 3:
            topo_ef = topo_ef.unsqueeze(0).expand(B, -1, -1, -1)
        if topo_fe.dim() == 3:
            topo_fe = topo_fe.unsqueeze(0).expand(B, -1, -1, -1)
        cross = torch.cat([topo_ef, topo_fe], dim=1)

        if topo_ee is not None and topo_ff is not None:
            if topo_ee.dim() == 3:
                topo_ee = topo_ee.unsqueeze(0).expand(B, -1, -1, -1)
            if topo_ff.dim() == 3:
                topo_ff = topo_ff.unsqueeze(0).expand(B, -1, -1, -1)
            uni = torch.cat([topo_ee, topo_ff], dim=1)
        else:
            uni = cross

        h = torch.cat([self.cross_enc(cross), self.uni_enc(uni)], dim=1)
        return self.proj(h)

class TopologySpatialGate(nn.Module):
    """利用 16x16 距离相关平面 (topo_ef, topo_fe) 将 30 通道 EEG 特征物理映射至 36 通道 fNIRS 空间"""
    def __init__(self, d_model=32):
        super().__init__()
        self.q_net = nn.Sequential(nn.Conv2d(36, d_model, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(1))
        self.k_net = nn.Sequential(nn.Conv2d(30, d_model, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(1))
        self.base_W = nn.Parameter(torch.randn(1, 36, 30) * 0.05)

    def forward(self, e_30, topo_ef, topo_fe):
        # e_30: (B, 30, L)
        tf_fe = topo_fe.unsqueeze(0) if topo_fe.dim() == 3 else topo_fe[:1]
        tf_ef = topo_ef.unsqueeze(0) if topo_ef.dim() == 3 else topo_ef[:1]
        # 通过平面相关性计算 (36, 30) 空间耦合权重
        corr_w = torch.einsum("bchw,bdhw->bcd", tf_fe, tf_ef) / 16.0  # (1, 36, 30)
        W = F.softmax(corr_w + self.base_W, dim=-1)
        return torch.matmul(W, e_30)  # (B, 36, L)

class SCDMUnet(nn.Module):
    """
    拓扑瓶颈先验引导型 SCDM (防止训练集死记硬背，强迫跨试次泛化):
    1. forward_prior: 30通道 EEG -> 拓扑空间门控映射 (36通道) -> 低秩瓶颈时序滤波 (32通道 + Dropout)
    2. forward: 残差扩散去噪器
    """
    def __init__(self, time_dim: int = 128, base_channels: int = 128, n_blocks: int = 6):
        super().__init__()
        c = base_channels
        self.time_dim = time_dim

        # 1. 拓扑引导低秩先验分支 (参数量精简 90%，彻底杜绝哈希记忆过拟合)
        self.topo_gate = TopologySpatialGate(d_model=32)
        self.prior_net = nn.Sequential(
            nn.Conv1d(36 + 30, 48, kernel_size=9, padding=4),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Conv1d(48, 48, kernel_size=9, padding=8, dilation=2, groups=6),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Conv1d(48, 36, kernel_size=7, padding=3)
        )
        # 可学习的群体血流动力学基线包络锚点 (1, 36, 256)
        self.hrf_template = nn.Parameter(torch.zeros(1, 36, 256))

        # 2. 扩散去噪精修分支
        self.eeg_hrf_enc = nn.Sequential(
            nn.Conv1d(30, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Dropout(0.2),
            nn.Conv1d(c, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
        )
        self.eeg_raw_enc = nn.Sequential(
            nn.Conv1d(30, c, 7, stride=4, padding=3), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv1d(c, c, 7, stride=4, padding=3), nn.GroupNorm(8, c), nn.SiLU(),
        )
        self.planes_enc = PlanesEncoder(cond_dim=time_dim)
        self.fnirs_proj = nn.Conv1d(36, c, 1)
        self.fuse = nn.Conv1d(c * 2, c, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.blocks = nn.ModuleList([ResBlock(c, time_dim, dropout=0.2) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, 36, 1))

    def encode_eeg(self, e_0, target_len=256):
        if e_0.size(-1) == 256:
            return self.eeg_hrf_enc(e_0)
        e = self.eeg_raw_enc(e_0)
        return F.interpolate(e, size=target_len, mode="linear", align_corners=False)

    def forward_prior(self, e_0, topo_ef, topo_fe, topo_ee=None, topo_ff=None):
        if e_0.size(-1) != 256:
            e_in = F.interpolate(e_0, size=256, mode="linear", align_corners=False)
        else:
            e_in = e_0

        # 训练时对条件 EEG 施加微扰与通道掩码，阻断死记硬背，提升未见样本 Sample PCC
        if self.training:
            e_in = e_in + 0.15 * torch.randn_like(e_in)

        e_mapped = self.topo_gate(e_in, topo_ef, topo_fe)  # (B, 36, 256)
        f_prior = self.prior_net(torch.cat([e_mapped, e_in], dim=1)) + self.hrf_template
        # 严格保持任务前 [-5s, 0s] (前50点) 基线归零
        f_prior = f_prior - f_prior[:, :, :50].mean(dim=-1, keepdim=True)
        return f_prior

    def forward(self, e_0, f_t, t, topo_ef, topo_fe, topo_ee=None, topo_ff=None):
        B = e_0.size(0)
        e = self.encode_eeg(e_0, f_t.size(-1))
        f = self.fnirs_proj(f_t)
        x = self.fuse(torch.cat([e, f], dim=1))

        t_emb = self.time_mlp(sinusoidal_embed(t, self.time_dim))
        p_emb = self.planes_enc(topo_ef, topo_fe, topo_ee, topo_ff, B=B)
        cond = t_emb + p_emb

        for block in self.blocks:
            x = block(x, cond)
        return self.out(x)