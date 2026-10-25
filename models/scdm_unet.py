import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.time_embedding import TimeEmbed
from models.modules.alignment import InputAlignment
from models.modules.scg_module import SCGModule
from models.modules.mtr_module import MTRModule

class SampleBlock(nn.Module):
    def __init__(self, c_in, c_out, seq_in, time_dim=64, mode="downsample"):
        super().__init__()
        self.mode = mode
        seq_out = seq_in // 2 if mode == "downsample" else seq_in * 2

        self.time_proj = nn.Linear(time_dim, c_in)

        self.scg1 = SCGModule(c_in, c_out, topo_in_ch=30, topo_out_ch=36)
        self.mtr1 = MTRModule(c_out, seq_in, mode=mode)
        self.scg2 = SCGModule(c_out, c_out, topo_in_ch=36, topo_out_ch=36)
        self.mtr2 = MTRModule(c_out, seq_out, mode="keep")
        self.scg3 = SCGModule(c_out, c_out, topo_in_ch=36, topo_out_ch=36)

        self.res_conv = nn.Conv1d(c_in, c_out, kernel_size=1)
        self.out_norm = nn.GroupNorm(min(4, c_out), c_out)

    def forward(self, e_feat, f_feat, t_emb, topo_ef, topo_fe):
        t_bias = self.time_proj(t_emb).unsqueeze(-1)
        x = f_feat + e_feat + t_bias

        scale = 0.5 if self.mode == "downsample" else 2.0
        res = F.interpolate(self.res_conv(x), scale_factor=scale, mode='linear', align_corners=False)

        h1 = self.scg1(x, topo_ef, topo_fe)
        h2 = self.mtr1(h1)
        h3 = self.scg2(h2, topo_fe, topo_fe) + h2
        h4 = self.mtr2(h3)
        h5 = self.scg3(h4, topo_fe, topo_fe)
        return self.out_norm(h5 + res)

class SCDMUnet(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_embed = TimeEmbed(time_dim)
        self.align = InputAlignment()

        # 编码器: 3个下采样 Sample Blocks
        self.block1 = SampleBlock(32, 16, seq_in=256, time_dim=time_dim, mode="downsample")
        self.block2 = SampleBlock(16, 8,  seq_in=128, time_dim=time_dim, mode="downsample")
        self.block3 = SampleBlock(8,  4,  seq_in=64,  time_dim=time_dim, mode="downsample")

        # 解码器: 3个上采样 Sample Blocks
        self.block4 = SampleBlock(4,  8,  seq_in=32,  time_dim=time_dim, mode="upsample")
        self.block5 = SampleBlock(8,  16, seq_in=64,  time_dim=time_dim, mode="upsample")
        self.block6 = SampleBlock(16, 32, seq_in=128, time_dim=time_dim, mode="upsample")

        # EEG 多尺度引导投影
        self.e_down1 = nn.Conv1d(32, 16, kernel_size=2, stride=2)
        self.e_down2 = nn.Conv1d(16, 8,  kernel_size=2, stride=2)
        self.e_down3 = nn.Conv1d(8,  4,  kernel_size=2, stride=2)

        # 全分辨率 (256) 融合输出头：结合 u3, f0, e0 与原始带噪输入 f_t
        self.final_head = nn.Sequential(
            nn.Conv1d(32 * 3 + 36, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 36, kernel_size=1)
        )

    def forward(self, e_t, f_t, t, topo_ef, topo_fe):
        t_emb = self.time_embed(t)
        e0, f0 = self.align.forward_align(e_t, f_t)  # [B, 32, 256]

        e1 = self.e_down1(e0)
        e2 = self.e_down2(e1)
        e3 = self.e_down3(e2)

        d1 = self.block1(e0, f0, t_emb, topo_ef, topo_fe)
        d2 = self.block2(e1, d1, t_emb, topo_ef, topo_fe)
        d3 = self.block3(e2, d2, t_emb, topo_ef, topo_fe)

        u1 = self.block4(e3, d3, t_emb, topo_ef, topo_fe)
        u2 = self.block5(e2, u1 + d2, t_emb, topo_ef, topo_fe)
        u3 = self.block6(e1, u2 + d1, t_emb, topo_ef, topo_fe)

        # 在 256 分辨率拼接跳跃连接，直接保留高频局部信息
        full_feat = torch.cat([u3, f0, e0, f_t], dim=1)
        pred_noise = self.final_head(full_feat)
        return pred_noise