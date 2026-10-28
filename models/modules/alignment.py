import torch
import torch.nn as nn
import torch.nn.functional as F

class InputAlignment(nn.Module):
    """原论文共享表征映射层: 将 EEG 与 fNIRS 统一映射至 (B, 32, 256) 表征空间"""
    def __init__(self, out_ch=32):
        super().__init__()
        self.out_ch = out_ch
        self.eeg_hrf_conv = nn.Sequential(
            nn.Conv1d(30, out_ch, kernel_size=5, padding=2),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
        self.eeg_raw_conv = nn.Sequential(
            nn.Conv1d(30, out_ch, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
        self.fnirs_conv = nn.Sequential(
            nn.Conv1d(36, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )

    def forward_align(self, e_t, f_t):
        if e_t.size(-1) == 256:
            e_feat = self.eeg_hrf_conv(e_t)
        else:
            e_h = self.eeg_raw_conv(e_t)
            e_feat = F.interpolate(e_h, size=256, mode='linear', align_corners=False)
        f_feat = self.fnirs_conv(f_t)
        return e_feat, f_feat