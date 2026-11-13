import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.time_embedding import TimeEmbed
from models.modules.alignment import InputAlignment
from models.modules.scg_module import SCGModule
from models.modules.mtr_module import MTRModule

class NativeSelfAttention(nn.Module):
    """消融基线 ATTN: 原生 DDPM 自注意力模块 (不使用 16x16 空间拓扑矩阵)"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj_in = nn.Conv1d(d_in, d_out, kernel_size=1) if d_in != d_out else nn.Identity()
        self.q = nn.Conv1d(d_out, d_out, kernel_size=1)
        self.k = nn.Conv1d(d_out, d_out, kernel_size=1)
        self.v = nn.Conv1d(d_out, d_out, kernel_size=1)
        num_groups = min(4, d_out) if d_out % min(4, d_out) == 0 else 1
        self.norm = nn.GroupNorm(num_groups, d_out)

    def forward(self, x):
        h = self.proj_in(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(h.size(-1)), dim=-1)
        return F.gelu(self.norm(torch.bmm(attn, v) + h))

class StandardConvBlock(nn.Module):
    """消融基线 COV: 原生 DDPM 标准单尺度一维卷积模块"""
    def __init__(self, channels, mode="downsample"):
        super().__init__()
        if mode == "downsample":
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        elif mode == "upsample":
            self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        num_groups = min(4, channels) if channels % min(4, channels) == 0 else 1
        self.norm = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))

class SCDMSampleBlock(nn.Module):
    """
    严格对齐原论文图文的 U-Net 采样模块 (Sample Block):
    每个采样模块内部包含【两个 SCG 和两个 MTR 交替处理数据】:
      交替阶段 1: SCG_1 (c_in -> c_out) -> MTR_1 (尺度变换 down/up) + 块内跳跃连接 1
      交替阶段 2: SCG_2 (c_out -> c_out) -> MTR_2 (等长双峰精炼 keep) + 块内跳跃连接 2
    """
    def __init__(self, c_in, c_out, seq_in, time_dim=64, mode="downsample", variant="SCG_EEG_MTR"):
        super().__init__()
        self.mode = mode
        self.variant = variant
        self.time_proj = nn.Linear(time_dim, c_in)

        topo_in = 36 if variant == "SCG_fNIRS_MTR" else 30
        topo_out = 36

        # 交替阶段 1: SCG_1 + MTR_1
        if variant in ("ATTN_COV", "ATTN_MTR"):
            self.spatial_1 = NativeSelfAttention(c_in, c_out)
            self.use_scg = False
        else:
            self.spatial_1 = SCGModule(c_in, c_out, topo_in_ch=topo_in, topo_out_ch=topo_out)
            self.use_scg = True

        if variant in ("ATTN_COV", "SCG_EEG_COV"):
            self.temporal_1 = StandardConvBlock(c_out, mode=mode)
        else:
            self.temporal_1 = MTRModule(c_out, seq_in, mode=mode)

        # 交替阶段 2: SCG_2 + MTR_2
        seq_out = seq_in // 2 if mode == "downsample" else (seq_in * 2 if mode == "upsample" else seq_in)
        if variant in ("ATTN_COV", "ATTN_MTR"):
            self.spatial_2 = NativeSelfAttention(c_out, c_out)
        else:
            self.spatial_2 = SCGModule(c_out, c_out, topo_in_ch=topo_in, topo_out_ch=topo_out)

        if variant in ("ATTN_COV", "SCG_EEG_COV"):
            self.temporal_2 = StandardConvBlock(c_out, mode="keep")
        else:
            self.temporal_2 = MTRModule(c_out, seq_out, mode="keep")

        self.res_conv1 = nn.Conv1d(c_in, c_out, kernel_size=1)
        self.res_conv2 = nn.Conv1d(c_out, c_out, kernel_size=1)
        num_groups = min(4, c_out) if c_out % min(4, c_out) == 0 else 1
        self.norm1 = nn.GroupNorm(num_groups, c_out)
        self.norm2 = nn.GroupNorm(num_groups, c_out)

    def forward(self, x, e_cond, t_emb, topo_q, topo_k):
        h_in = x + e_cond + self.time_proj(t_emb).unsqueeze(-1)
        if self.mode == "downsample":
            res1 = F.avg_pool1d(self.res_conv1(h_in), kernel_size=2, stride=2)
        elif self.mode == "upsample":
            res1 = F.interpolate(self.res_conv1(h_in), scale_factor=2.0, mode="linear", align_corners=False)
        else:
            res1 = self.res_conv1(h_in)

        # 第 1 次交替: SCG_1 -> MTR_1 + Skip 1
        h1 = self.spatial_1(h_in, topo_q, topo_k) if self.use_scg else self.spatial_1(h_in)
        h1 = self.temporal_1(h1)
        h1 = self.norm1(h1 + res1)

        # 第 2 次交替: SCG_2 -> MTR_2 + Skip 2
        res2 = self.res_conv2(h1)
        h2 = self.spatial_2(h1, topo_q, topo_k) if self.use_scg else self.spatial_2(h1)
        h2 = self.temporal_2(h2)
        return self.norm2(h2 + res2)

class SCDMUnet(nn.Module):
    """
    原论文 6-Block SCDM U-Net 去噪主干网络:
    - 修复 HbR 与 HbO 在入口处的反相抵消问题: 入口处将 HbO 极性对齐后与 HbR 同相融合编码，出口处独立解码
    - 每个 Sample Block 包含两个 SCG 与两个 MTR 交替处理及跳跃连接
    """
    def __init__(self, time_dim: int = 64, base_channels: int = 32, n_blocks: int = 6,
                 out_channels: int = 72, variant: str = "SCG_EEG_MTR"):
        super().__init__()
        assert n_blocks == 6, "原论文规定必须为 6 个 Sample Blocks (3 Down + 3 Up)"
        self.out_channels = out_channels
        self.variant = variant
        c = base_channels  # 32
        self.time_embed = TimeEmbed(dim=time_dim)
        self.align = InputAlignment(out_ch=c)

        # 同相增强适配器: 避免 HbR 与 HbO 正负直接相加抵消
        self.ft_in_adapter = nn.Conv1d(72, 36, kernel_size=3, padding=1, groups=36) if out_channels == 72 else nn.Identity()
        self.fnirs_cond_adapter = nn.Conv1d(72, 30, kernel_size=1)

        self.noise_gate = nn.Sequential(
            nn.Linear(time_dim, c),
            nn.Sigmoid()
        )

        self.e_proj_d1 = nn.Identity()                                      # (32, 256)
        self.e_proj_d2 = nn.Conv1d(c, c // 2, kernel_size=2, stride=2)      # (16, 128)
        self.e_proj_d3 = nn.Conv1d(c // 2, c // 4, kernel_size=2, stride=2) # (8, 64)
        self.e_proj_u1 = nn.Conv1d(c // 4, c // 8, kernel_size=2, stride=2) # (4, 32)

        self.block1_down = SCDMSampleBlock(c,      c // 2, seq_in=256, time_dim=time_dim, mode="downsample", variant=variant)
        self.block2_down = SCDMSampleBlock(c // 2, c // 4, seq_in=128, time_dim=time_dim, mode="downsample", variant=variant)
        self.block3_down = SCDMSampleBlock(c // 4, c // 8, seq_in=64,  time_dim=time_dim, mode="downsample", variant=variant)

        self.block4_up   = SCDMSampleBlock(c // 8, c // 4, seq_in=32,  time_dim=time_dim, mode="upsample", variant=variant)
        self.block5_up   = SCDMSampleBlock(c // 4, c // 2, seq_in=64,  time_dim=time_dim, mode="upsample", variant=variant)
        self.block6_up   = SCDMSampleBlock(c // 2, c,      seq_in=128, time_dim=time_dim, mode="upsample", variant=variant)

        self.leadfield_hbr = nn.Sequential(
            nn.Conv1d(30, 36, kernel_size=5, padding=2),
            nn.GroupNorm(6, 36),
            nn.GELU(),
            nn.Conv1d(36, 36, kernel_size=1)
        )
        self.leadfield_hbo = nn.Sequential(
            nn.Conv1d(30, 36, kernel_size=5, padding=2),
            nn.GroupNorm(6, 36),
            nn.GELU(),
            nn.Conv1d(36, 36, kernel_size=1)
        )

        self.time_out = nn.Linear(time_dim, c + 36)
        self.refine_hbr = nn.Sequential(
            nn.Conv1d(c + 36, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 36, kernel_size=3, padding=1)
        )
        self.refine_hbo = nn.Sequential(
            nn.Conv1d(c + 36, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 36, kernel_size=3, padding=1)
        )

    def forward(self, e_0, f_t, t, topo_ef, topo_fe, corr_ef=None, spatial_affinity=None, topo_ff=None, f_self_cond=None):
        t_emb = self.time_embed(t)
        if f_t.size(1) == 72:
            # 将 HbR (前36ch) 与翻转极性后的 -HbO (后36ch) 按通道配对组合，彻底消除反相抵消!
            f_pair = torch.stack([f_t[:, :36, :], -f_t[:, 36:, :]], dim=2).reshape(f_t.size(0), 72, f_t.size(-1))
            f_t_36 = 0.5 * (f_t[:, :36, :] - f_t[:, 36:, :]) + 0.5 * self.ft_in_adapter(f_pair)
        else:
            f_t_36 = f_t

        if self.variant == "SCG_fNIRS_MTR" and f_self_cond is not None:
            cond_in = self.fnirs_cond_adapter(f_self_cond) if f_self_cond.size(1) == 72 else f_self_cond[:, :30, :]
        else:
            cond_in = e_0

        e0, f0 = self.align.forward_align(cond_in, f_t_36)
        g_t = self.noise_gate(t_emb).unsqueeze(-1)
        f_in = g_t * f0 + (1.0 - g_t) * e0

        ed1 = self.e_proj_d1(e0)
        ed2 = self.e_proj_d2(ed1)
        ed3 = self.e_proj_d3(ed2)
        eu1 = self.e_proj_u1(ed3)

        if self.variant == "SCG_fNIRS_MTR":
            tq = topo_ff if topo_ff is not None else topo_fe
            tk = topo_ff if topo_ff is not None else topo_fe
        else:
            tq, tk = topo_ef, topo_fe

        s1 = self.block1_down(f_in, ed1, t_emb, tq, tk)
        s2 = self.block2_down(s1,   ed2, t_emb, tq, tk)
        s3 = self.block3_down(s2,   ed3, t_emb, tq, tk)

        u4 = self.block4_up(s3,      eu1, t_emb, tq, tk)
        u5 = self.block5_up(u4 + s2, ed3, t_emb, tq, tk)
        u6 = self.block6_up(u5 + s1, ed2, t_emb, tq, tk)

        c_256 = cond_in if cond_in.size(-1) == 256 else F.interpolate(cond_in, size=256, mode="linear", align_corners=False)
        if corr_ef is not None and self.variant in ("SCG_EEG_MTR", "SCG_EEG_COV", "SCG_fNIRS_MTR"):
            w_prior = corr_ef.to(c_256.device).T
            if spatial_affinity is not None:
                w_prior = w_prior * spatial_affinity.to(c_256.device)
            w_prior = w_prior / (w_prior.sum(dim=-1, keepdim=True) + 1e-6)
            e_prior = torch.matmul(w_prior.unsqueeze(0), c_256)
            map_hbr = 0.55 * e_prior + 0.45 * self.leadfield_hbr(c_256)
            map_hbo = -0.55 * e_prior + 0.45 * self.leadfield_hbo(c_256)
        else:
            map_hbr = self.leadfield_hbr(c_256)
            map_hbo = self.leadfield_hbo(c_256)

        t_bias = self.time_out(t_emb).unsqueeze(-1)
        out_hbr = self.refine_hbr(torch.cat([u6 + e0, map_hbr], dim=1) + t_bias)
        if self.out_channels == 36:
            return out_hbr
        # 解码器出口将同相共享特征 u6 + e0 翻转符号送入 refine_hbo，保证 HbO 与 HbR 具备同等高信噪比!
        out_hbo = self.refine_hbo(torch.cat([-(u6 + e0), map_hbo], dim=1) + t_bias)
        return torch.cat([out_hbr, out_hbo], dim=1)