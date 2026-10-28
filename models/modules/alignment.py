import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputAlignment(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        self.out_ch = out_ch
        # 针对 256 点 HRF 脑电特征的编码器
        self.eeg_hrf_conv = nn.Sequential(
            nn.Conv1d(30, out_ch, kernel_size=5, padding=2),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
        # 针对 4000 点原始脑电特征的后备编码器
        self.eeg_raw_conv = nn.Sequential(
            nn.Conv1d(30, out_ch, kernel_size=7, stride=5, padding=3),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=3, padding=2),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
        # fNIRS 编码: [B, 36, 256] -> [B, out_ch, 256]
        self.fnirs_conv = nn.Sequential(
            nn.Conv1d(36, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
        # 任务相位位置编码投影 (锚定 [-5s, 20.6s] 的血流动力学响应时钟)
        self.phase_proj = nn.Conv1d(16, out_ch, kernel_size=1)
        self.register_buffer("phase_clock", self._build_phase_clock(256, 16))

    @staticmethod
    def _build_phase_clock(seq_len=256, dim=16):
        pos = torch.linspace(0.0, 1.0, seq_len).unsqueeze(0)  # (1, 256)
        freqs = torch.arange(1, dim // 2 + 1).float().unsqueeze(1) * math.pi
        sin_p = torch.sin(freqs * pos)
        cos_p = torch.cos(freqs * pos)
        return torch.cat([sin_p, cos_p], dim=0).unsqueeze(0)  # (1, 16, 256)

    def forward_align(self, e_t, f_t):
        if e_t.size(-1) == 256:
            e_feat = self.eeg_hrf_conv(e_t)
        else:
            e_h = self.eeg_raw_conv(e_t)
            e_feat = F.interpolate(e_h, size=256, mode='linear', align_corners=False)
        phase_emb = self.phase_proj(self.phase_clock)
        e_feat = e_feat + phase_emb
        f_feat = self.fnirs_conv(f_t) + phase_emb
        return e_feat, f_feat