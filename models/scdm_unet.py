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
    - 结合跨模态距离相关矩阵 C_ef 直接引导试次级神经血管映射
    """
    def __init__(self, time_dim: int = 64, base_channels: int = 32, n_blocks: int = 6):
        super().__init__()
        assert n_blocks == 6, "原论文 SCDM 规定必须为 6 个 Sample Blocks (3 Down + 3 Up)"
        c = base_channels  # 32
        self.time_embed = TimeEmbed(dim=time_dim)
        self.align = InputAlignment(out_ch=c)

        # EEG 多尺度条件金字塔投影 (匹配 6 个 Sample Block 的输入通道与分辨率)
        self.e_proj_d1 = nn.Identity()                                      # (32, 256)
        self.e_proj_d2 = nn.Conv1d(c, c // 2, kernel_size=2, stride=2)      # (16, 128)
        self.e_proj_d3 = nn.Conv1d(c // 2, c // 4, kernel_size=2, stride=2) # (8, 64)
        self.e_proj_u1 = nn.Conv1d(c // 4, c // 8, kernel_size=2, stride=2) # (4, 32)

        # 原论文 6 个 Sample Blocks:
        # Encoder: Block 1 (32->16, 256->128), Block 2 (16->8, 128->64), Block 3 (8->4, 64->32)
        self.block1_down = SCDMSampleBlock(c,      c // 2, seq_in=256, time_dim=time_dim, mode="downsample")
        self.block2_down = SCDMSampleBlock(c // 2, c // 4, seq_in=128, time_dim=time_dim, mode="downsample")
        self.block3_down = SCDMSampleBlock(c // 4, c // 8, seq_in=64,  time_dim=time_dim, mode="downsample")

        # Decoder: Block 4 (4->8, 32->64), Block 5 (8->16, 64->128), Block 6 (16->32, 128->256)
        self.block4_up   = SCDMSampleBlock(c // 8, c // 4, seq_in=32,  time_dim=time_dim, mode="upsample")
        self.block5_up   = SCDMSampleBlock(c // 4, c // 2, seq_in=64,  time_dim=time_dim, mode="upsample")
        self.block6_up   = SCDMSampleBlock(c // 2, c,      seq_in=128, time_dim=time_dim, mode="upsample")

        # 跨模态解剖导联映射分支 (保证单试次 EEG 偏侧化特征无损直达输出层)
        self.spatial_leadfield = nn.Conv1d(30, 36, kernel_size=1, bias=False)
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(c + 36, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 36, kernel_size=5, padding=2)
        )

    def forward(self, e_0, f_t, t, topo_ef, topo_fe, corr_ef=None):
        t_emb = self.time_embed(t)
        e0, f0 = self.align.forward_align(e_0, f_t)  # 共享表征 (B, 32, 256)

        # 构建多尺度 EEG 条件特征
        ed1 = self.e_proj_d1(e0)   # (B, 32, 256)
        ed2 = self.e_proj_d2(ed1)  # (B, 16, 128)
        ed3 = self.e_proj_d3(ed2)  # (B, 8, 64)
        eu1 = self.e_proj_u1(ed3)  # (B, 4, 32)

        # 6 个 Sample Blocks 编码-解码路径
        s1 = self.block1_down(f0, ed1, t_emb, topo_ef, topo_fe)       # (B, 16, 128)
        s2 = self.block2_down(s1, ed2, t_emb, topo_ef, topo_fe)       # (B, 8, 64)
        s3 = self.block3_down(s2, ed3, t_emb, topo_ef, topo_fe)       # (B, 4, 32)

        u4 = self.block4_up(s3,      eu1, t_emb, topo_ef, topo_fe)    # (B, 8, 64)
        u5 = self.block5_up(u4 + s2, ed3, t_emb, topo_ef, topo_fe)    # (B, 16, 128)
        u6 = self.block6_up(u5 + s1, ed2, t_emb, topo_ef, topo_fe)    # (B, 32, 256)

        # 结合跨模态通道映射输出最终去噪预测
        e_256 = e_0 if e_0.size(-1) == 256 else F.interpolate(e_0, size=256, mode="linear", align_corners=False)
        if corr_ef is not None:
            # 融合训练集距离相关矩阵 C_ef^T (36, 30) 与可学习导联权重
            c_mat = corr_ef.to(e_256.device)
            c_norm = c_mat.T / (c_mat.T.sum(dim=-1, keepdim=True) + 1e-6)
            e_mapped = 0.5 * torch.matmul(c_norm.unsqueeze(0), e_256) + 0.5 * self.spatial_leadfield(e_256)
        else:
            e_mapped = self.spatial_leadfield(e_256)

        out = self.temporal_refine(torch.cat([u6 + e0, e_mapped], dim=1))
        return out