import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.time_embedding import TimeEmbed
from models.modules.alignment import InputAlignment
from models.modules.scg_module import SCGModule
from models.modules.mtr_module import MTRModule

class SingleStreamRepBlock(nn.Module):
    """原论文单模态表征块：EEG流与fNIRS流共享相同架构，但权重独立训练"""
    def __init__(self, c_in, c_out, seq_in, topo_ch, time_dim=64, mode="downsample"):
        super().__init__()
        self.mode = mode
        seq_out = seq_in // 2 if mode == "downsample" else seq_in * 2
        self.time_proj = nn.Linear(time_dim, c_in)

        self.scg1 = SCGModule(c_in, c_out, topo_in_ch=topo_ch, topo_out_ch=topo_ch)
        self.mtr1 = MTRModule(c_out, seq_in, mode=mode)
        self.scg2 = SCGModule(c_out, c_out, topo_in_ch=topo_ch, topo_out_ch=topo_ch)
        self.mtr2 = MTRModule(c_out, seq_out, mode="keep")

        self.res_conv = nn.Conv1d(c_in, c_out, kernel_size=1)
        self.norm = nn.GroupNorm(min(4, c_out), c_out)

    def forward(self, x, t_emb, topo_self):
        h = x + self.time_proj(t_emb).unsqueeze(-1)
        scale = 0.5 if self.mode == "downsample" else 2.0
        res = F.interpolate(self.res_conv(h), scale_factor=scale, mode='linear', align_corners=False)

        h = self.scg1(h, topo_self, topo_self)
        h = self.mtr1(h)
        h = self.scg2(h, topo_self, topo_self)
        h = self.mtr2(h)
        return self.norm(h + res)

class DualStreamSampleBlock(nn.Module):
    """
    原论文双流耦合块:
    1. EEG 与 fNIRS 各自通过独立的 Representation Block
    2. EEG 输出不仅传给下一层，还通过跨模态 SCG 映射为 fNIRS 表征并做残差连接
    """
    def __init__(self, c_in, c_out, seq_in, time_dim=64, mode="downsample"):
        super().__init__()
        self.eeg_block = SingleStreamRepBlock(c_in, c_out, seq_in, topo_ch=30, time_dim=time_dim, mode=mode)
        self.fnirs_block = SingleStreamRepBlock(c_in, c_out, seq_in, topo_ch=36, time_dim=time_dim, mode=mode)
        # 跨模态 SCG 映射模块: 将 EEG 表征 (topo_ef) 映射至 fNIRS 表征空间 (topo_fe)
        self.cross_scg = SCGModule(c_out, c_out, topo_in_ch=30, topo_out_ch=36)
        self.out_norm = nn.GroupNorm(min(4, c_out), c_out)

    def forward(self, e_in, f_in, t_emb, topo_ef, topo_fe):
        e_out = self.eeg_block(e_in, t_emb, topo_ef)
        f_out = self.fnirs_block(f_in, t_emb, topo_fe)
        # 跨模态映射与残差连接
        e_to_f = self.cross_scg(e_out, topo_ef, topo_fe)
        f_fused = self.out_norm(f_out + e_to_f)
        return e_out, f_fused, e_to_f

class SCDMUnet(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_embed = TimeEmbed(time_dim)
        self.align = InputAlignment()

        # 3个下采样双流块 (256 -> 128 -> 64 -> 32)
        self.block1 = DualStreamSampleBlock(32, 16, seq_in=256, time_dim=time_dim, mode="downsample")
        self.block2 = DualStreamSampleBlock(16, 8,  seq_in=128, time_dim=time_dim, mode="downsample")
        self.block3 = DualStreamSampleBlock(8,  4,  seq_in=64,  time_dim=time_dim, mode="downsample")

        # 3个上采样双流块 (32 -> 64 -> 128 -> 256)
        self.block4 = DualStreamSampleBlock(4,  8,  seq_in=32,  time_dim=time_dim, mode="upsample")
        self.block5 = DualStreamSampleBlock(8,  16, seq_in=64,  time_dim=time_dim, mode="upsample")
        self.block6 = DualStreamSampleBlock(16, 32, seq_in=128, time_dim=time_dim, mode="upsample")

        # 初始对齐层的跨模态 SCG
        self.align_cross_scg = SCGModule(32, 32, topo_in_ch=30, topo_out_ch=36)

        # 汇总 6 个连接层的结果并输出 [B, 36, 256]
        total_ch = 32 + 32 + (16 + 8 + 4 + 8 + 16 + 32)
        self.final_head = nn.Sequential(
            nn.Conv1d(total_ch, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 36, kernel_size=5, padding=2)
        )

    def forward(self, e_t, f_t, t, topo_ef, topo_fe):
        t_emb = self.time_embed(t)
        e0, f0 = self.align.forward_align(e_t, f_t)
        e0_mapped = self.align_cross_scg(e0, topo_ef, topo_fe)

        e1, f1, c1 = self.block1(e0, f0 + e0_mapped, t_emb, topo_ef, topo_fe)
        e2, f2, c2 = self.block2(e1, f1, t_emb, topo_ef, topo_fe)
        e3, f3, c3 = self.block3(e2, f2, t_emb, topo_ef, topo_fe)

        e4, f4, c4 = self.block4(e3, f3, t_emb, topo_ef, topo_fe)
        e5, f5, c5 = self.block5(e4 + e2, f4 + f2, t_emb, topo_ef, topo_fe)
        e6, f6, c6 = self.block6(e5 + e1, f5 + f1, t_emb, topo_ef, topo_fe)

        # 将 6 个层级的跨模态连接结果统一上采样至 256 并融合 (对应原论文 "The results of these six connections are ultimately processed")
        multi_scale = [
            f6, e0_mapped,
            F.interpolate(c1, size=256, mode='linear', align_corners=False),
            F.interpolate(c2, size=256, mode='linear', align_corners=False),
            F.interpolate(c3, size=256, mode='linear', align_corners=False),
            F.interpolate(c4, size=256, mode='linear', align_corners=False),
            F.interpolate(c5, size=256, mode='linear', align_corners=False),
            c6
        ]
        pred_f0 = self.final_head(torch.cat(multi_scale, dim=1))
        # 强制输出满足任务前基线归零的物理先验
        pred_f0 = pred_f0 - pred_f0[:, :, :50].mean(dim=-1, keepdim=True)
        return pred_f0