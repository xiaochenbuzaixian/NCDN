import torch
import torch.nn as nn
import torch.nn.functional as F

class InputAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        # EEG: [B, 30, 4000] -> 保持严格时序对齐降采样至 [B, 32, 256]
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(30, 32, kernel_size=7, stride=5, padding=3),  # 4000 -> 800
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=3, padding=2),  # 800 -> 267
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        self.eeg_refine = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

        # fNIRS: [B, 36, 256] -> [B, 32, 256] (严格保留逐时间点局部对应关系)
        self.fnirs_conv = nn.Sequential(
            nn.Conv1d(36, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

    def forward_align(self, e_t, f_t):
        e_h = self.eeg_conv(e_t)
        e_h = F.interpolate(e_h, size=256, mode='linear', align_corners=False)
        e_feat = self.eeg_refine(e_h)
        f_feat = self.fnirs_conv(f_t)
        return e_feat, f_feat