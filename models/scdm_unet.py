import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.time_embedding import TimeEmbed
from models.modules.alignment import InputAlignment
from models.modules.scg_module import SCGModule
from models.modules.mtr_module import MTRModule

class SignedTopologyCoupling(nn.Module):
    """
    修复后的非零跨模态空间拓扑耦合层:
    利用 2D 空间卷积提取 16x16 平面特征 (解决互斥格点直接点乘为 0 的 Bug)，
    并结合正负可变的推挽映射与左右半球对称差分
    """
    def __init__(self):
        super().__init__()
        self.ef_conv = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(30, 30, kernel_size=3, padding=1)
        )
        self.fe_conv = nn.Sequential(
            nn.Conv2d(36, 36, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(36, 36, kernel_size=3, padding=1)
        )
        self.W_base = nn.Parameter(torch.randn(1, 36, 30) * 0.05)
        self.lat_conv = nn.Conv1d(30, 36, kernel_size=1, bias=False)

    def forward(self, e_30, topo_ef, topo_fe):
        t_ef = topo_ef.unsqueeze(0) if topo_ef.dim() == 3 else topo_ef[:1]
        t_fe = topo_fe.unsqueeze(0) if topo_fe.dim() == 3 else topo_fe[:1]

        # 通过 5x5 与 3x3 感受野弥合 16x16 网格上的相邻电极与光极，获得非零空间耦合矩阵
        f_ef = self.ef_conv(t_ef).flatten(2)  # (1, 30, 256)
        f_fe = self.fe_conv(t_fe).flatten(2)  # (1, 36, 256)
        corr_w = torch.bmm(f_fe, f_ef.transpose(1, 2)) / 16.0  # (1, 36, 30)

        W = torch.tanh(corr_w + self.W_base) / math.sqrt(30.0)
        e_lat = e_30 - torch.flip(e_30, dims=[1])
        return torch.matmul(W, e_30) + 0.25 * self.lat_conv(e_lat)

class SCDMSampleBlock(nn.Module):
    """原论文 Sample Block: 融合 SCG 空间跨模态注意力与 MTR 多尺度时序表示"""
    def __init__(self, c_in, c_out, seq_in, time_dim=128, mode="downsample"):
        super().__init__()
        self.mode = mode
        self.time_proj = nn.Linear(time_dim, c_in)
        self.scg = SCGModule(c_in, c_out, topo_in_ch=30, topo_out_ch=36)
        self.mtr = MTRModule(c_out, seq_in, mode=mode)
        self.res_conv = nn.Conv1d(c_in, c_out, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, c_out), c_out)

    def forward(self, x, e_guide, t_emb, topo_ef, topo_fe):
        h = x + e_guide + self.time_proj(t_emb).unsqueeze(-1)
        if self.mode == "downsample":
            res = F.avg_pool1d(self.res_conv(h), kernel_size=2, stride=2)
        elif self.mode == "upsample":
            res = F.interpolate(self.res_conv(h), scale_factor=2.0, mode="linear", align_corners=False)
        else:
            res = self.res_conv(h)

        h = self.scg(h, topo_ef, topo_fe)
        h = self.mtr(h)
        return self.norm(h + res)

class SCDMUnet(nn.Module):
    """
    完整复现原论文的 Prior-Guided SCDM U-Net:
    1. forward_prior: 结合任务相位时钟与 SCG 拓扑映射生成单试次血流动力学先验 (全时段零均值，与 preprocess 严格一致)
    2. forward: 采用 InputAlignment + SCG + MTR 构建多尺度编码-解码去噪器
    """
    def __init__(self, time_dim: int = 128, base_channels: int = 64, n_blocks: int = 6):
        super().__init__()
        c = base_channels
        self.time_embed = TimeEmbed(dim=time_dim)
        self.align = InputAlignment(out_ch=c)

        # 1. 神经血管耦合先验分支 (含任务相位时钟编码，解决未见样本相位迷失)
        self.topo_coupling = SignedTopologyCoupling()
        self.prior_scg = SCGModule(36 + 30 + 16, 64, topo_in_ch=30, topo_out_ch=36)
        self.prior_mtr = MTRModule(64, seq_len=256, mode="keep")
        self.prior_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, padding=6, dilation=2, groups=8),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Conv1d(64, 36, kernel_size=5, padding=2)
        )

        # 2. 原论文多尺度 U-Net (SCG + MTR) 去噪主干
        self.e_down1 = nn.AvgPool1d(2, 2)  # 256 -> 128
        self.e_down2 = nn.AvgPool1d(2, 2)  # 128 -> 64

        self.down1 = SCDMSampleBlock(c, c, seq_in=256, time_dim=time_dim, mode="downsample")      # -> 128
        self.down2 = SCDMSampleBlock(c, c, seq_in=128, time_dim=time_dim, mode="downsample")      # -> 64
        self.mid   = SCDMSampleBlock(c, c, seq_in=64,  time_dim=time_dim, mode="keep")            # -> 64
        self.up1   = SCDMSampleBlock(c, c, seq_in=64,  time_dim=time_dim, mode="upsample")        # -> 128
        self.up2   = SCDMSampleBlock(c, c, seq_in=128, time_dim=time_dim, mode="upsample")        # -> 256

        self.out_head = nn.Sequential(
            nn.Conv1d(c * 2, c, kernel_size=3, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv1d(c, 36, kernel_size=1)
        )

    def forward_prior(self, e_0, topo_ef, topo_fe):
        if e_0.size(-1) != 256:
            e_in = F.interpolate(e_0, size=256, mode="linear", align_corners=False)
        else:
            e_in = e_0

        B = e_in.size(0)
        e_mapped = self.topo_coupling(e_in, topo_ef, topo_fe)  # (B, 36, 256)
        phase_clk = self.align.phase_clock.expand(B, -1, -1)   # (B, 16, 256)

        h = torch.cat([e_mapped, e_in, phase_clk], dim=1)      # (B, 82, 256)
        h = self.prior_scg(h, topo_ef, topo_fe)                # (B, 64, 256)
        h = h + self.prior_mtr(h)                              # (B, 64, 256)
        f_prior = self.prior_head(h)                           # (B, 36, 256)

        # 关键修复：与 preprocess_shin.py 的全时段零均值完全对齐，消除直流偏移死锁！
        return f_prior - f_prior.mean(dim=-1, keepdim=True)

    def forward(self, e_0, f_t, t, topo_ef, topo_fe, *args):
        t_emb = self.time_embed(t)
        e0, f0 = self.align.forward_align(e_0, f_t)  # (B, 64, 256)

        e1 = self.e_down1(e0)  # (B, 64, 128)
        e2 = self.e_down2(e1)  # (B, 64, 64)

        d1 = self.down1(f0, e0, t_emb, topo_ef, topo_fe)       # (B, 64, 128)
        d2 = self.down2(d1, e1, t_emb, topo_ef, topo_fe)       # (B, 64, 64)
        m  = self.mid(d2, e2, t_emb, topo_ef, topo_fe)         # (B, 64, 64)
        u1 = self.up1(m + d2, e2, t_emb, topo_ef, topo_fe)     # (B, 64, 128)
        u2 = self.up2(u1 + d1, e1, t_emb, topo_ef, topo_fe)    # (B, 64, 256)

        return self.out_head(torch.cat([u2, f0], dim=1))