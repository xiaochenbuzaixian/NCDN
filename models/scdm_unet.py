import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.time_embedding import TimeEmbed
from models.modules.alignment import InputAlignment
from models.modules.scg_module import SCGModule
from models.modules.mtr_module import MTRModule

class SCDMSampleBlock(nn.Module):
    """原论文 Sample Block: 包含 SCG (改变通道数) 与 MTR (改变序列长度)"""
    def __init__(self, c_in, c_out, seq_in, time_dim=64, mode="downsample"):
        super().__init__()
        self.mode = mode
        self.time_proj = nn.Linear(time_dim, c_in)
        self.scg = SCGModule(c_in, c_out, topo_in_ch=30, topo_out_ch=36)
        self.mtr = MTRModule(c_out, seq_in, mode=mode)
        self.res_conv = nn.Conv1d(c_in, c_out, kernel_size=1)
        num_groups = min(4, c_out) if c_out % min(4, c_out) == 0 else 1
        self.norm = nn.GroupNorm(num_groups, c_out)

    def forward(self, x, e_cond, t_emb, topo_ef, topo_fe):
        h = x + e_cond + self.time_proj(t_emb).unsqueeze(-1)
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
    严格对齐原论文的 6-Block SCDM U-Net:
    - 统一共享表征输入: (B, 32, 256)
    - 3 个编码 Sample Blocks: (32,256) -> (16,128) -> (8,64) -> (4,32)
    - 3 个解码 Sample Blocks: (4,32)  -> (8,64)  -> (16,128) -> (32,256)
    - 引入时间步自适应门控 (Time-Aware Input Gating)，抑制 t->T 时纯高斯噪声对试次级 EEG 相位的破坏
    """
    def __init__(self, time_dim: int = 64, base_channels: int = 32, n_blocks: int = 6):
        super().__init__()
        assert n_blocks == 6, "原论文 SCDM 规定必须为 6 个 Sample Blocks (3 Down + 3 Up)"
        c = base_channels  # 32
        self.time_embed = TimeEmbed(dim=time_dim)
        self.align = InputAlignment(out_ch=c)

        # 时间步自适应输入调制门 (根据时间步 t 动态平衡加噪状态 f0 与条件表征 e0)
        self.noise_gate = nn.Sequential(
            nn.Linear(time_dim, c),
            nn.Sigmoid()
        )

        # EEG 多尺度条件金字塔投影
        self.e_proj_d1 = nn.Identity()                                      # (32, 256)
        self.e_proj_d2 = nn.Conv1d(c, c // 2, kernel_size=2, stride=2)      # (16, 128)
        self.e_proj_d3 = nn.Conv1d(c // 2, c // 4, kernel_size=2, stride=2) # (8, 64)
        self.e_proj_u1 = nn.Conv1d(c // 4, c // 8, kernel_size=2, stride=2) # (4, 32)

        # 原论文 6 个 Sample Blocks:
        self.block1_down = SCDMSampleBlock(c,      c // 2, seq_in=256, time_dim=time_dim, mode="downsample")
        self.block2_down = SCDMSampleBlock(c // 2, c // 4, seq_in=128, time_dim=time_dim, mode="downsample")
        self.block3_down = SCDMSampleBlock(c // 4, c // 8, seq_in=64,  time_dim=time_dim, mode="downsample")

        self.block4_up   = SCDMSampleBlock(c // 8, c // 4, seq_in=32,  time_dim=time_dim, mode="upsample")
        self.block5_up   = SCDMSampleBlock(c // 4, c // 2, seq_in=64,  time_dim=time_dim, mode="upsample")
        self.block6_up   = SCDMSampleBlock(c // 2, c,      seq_in=128, time_dim=time_dim, mode="upsample")

        # 局部神经血管导联映射 (融合解剖距离亲和度与跨通道偏侧化对比)
        self.spatial_leadfield = nn.Sequential(
            nn.Conv1d(30, 36, kernel_size=5, padding=2),
            nn.GroupNorm(6, 36),
            nn.GELU(),
            nn.Conv1d(36, 36, kernel_size=1)
        )
        self.time_out = nn.Linear(time_dim, c + 36)
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(c + 36, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 36, kernel_size=3, padding=1)
        )

    def forward(self, e_0, f_t, t, topo_ef, topo_fe, corr_ef=None, spatial_affinity=None):
        t_emb = self.time_embed(t)
        e0, f0 = self.align.forward_align(e_0, f_t)  # (B, 32, 256)

        # 根据时间步 t 调制输入噪声特征 f0，防止高噪声步随机高斯波动掩盖单试次 EEG 偏侧化特征
        g_t = self.noise_gate(t_emb).unsqueeze(-1)
        f_in = g_t * f0 + (1.0 - g_t) * e0

        ed1 = self.e_proj_d1(e0)   # (B, 32, 256)
        ed2 = self.e_proj_d2(ed1)  # (B, 16, 128)
        ed3 = self.e_proj_d3(ed2)  # (B, 8, 64)
        eu1 = self.e_proj_u1(ed3)  # (B, 4, 32)

        s1 = self.block1_down(f_in, ed1, t_emb, topo_ef, topo_fe)
        s2 = self.block2_down(s1,   ed2, t_emb, topo_ef, topo_fe)
        s3 = self.block3_down(s2,   ed3, t_emb, topo_ef, topo_fe)

        u4 = self.block4_up(s3,      eu1, t_emb, topo_ef, topo_fe)
        u5 = self.block5_up(u4 + s2, ed3, t_emb, topo_ef, topo_fe)
        u6 = self.block6_up(u5 + s1, ed2, t_emb, topo_ef, topo_fe)

        e_256 = e_0 if e_0.size(-1) == 256 else F.interpolate(e_0, size=256, mode="linear", align_corners=False)
        if corr_ef is not None:
            w_prior = corr_ef.to(e_256.device).T  # (36, 30)
            if spatial_affinity is not None:
                w_prior = w_prior * spatial_affinity.to(e_256.device)
            w_prior = w_prior / (w_prior.sum(dim=-1, keepdim=True) + 1e-6)
            e_mapped = 0.5 * torch.matmul(w_prior.unsqueeze(0), e_256) + 0.5 * self.spatial_leadfield(e_256)
        else:
            e_mapped = self.spatial_leadfield(e_256)

        h_final = torch.cat([u6 + e0, e_mapped], dim=1) + self.time_out(t_emb).unsqueeze(-1)
        return self.temporal_refine(h_final)