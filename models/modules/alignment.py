import torch
import torch.nn as nn
import torch.nn.functional as F

class InputAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        # 分支1: EEG 时域波形下采样 [B, 30, 4000] -> [B, 16, 267]
        self.eeg_wave_conv = nn.Sequential(
            nn.Conv1d(30, 32, kernel_size=7, stride=5, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=5, stride=3, padding=2),
            nn.GroupNorm(4, 16),
            nn.GELU()
        )

        # 分支2: EEG 节律能量包络 (ERD/ERS Power Envelope) [B, 30, 4000] -> [B, 16, 250]
        self.eeg_env_pool = nn.AvgPool1d(kernel_size=32, stride=16, padding=8)
        self.eeg_env_conv = nn.Sequential(
            nn.Conv1d(30, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU()
        )

        self.eeg_fuse = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

        # fNIRS 编码: [B, 36, 256] -> [B, 32, 256]
        self.fnirs_conv = nn.Sequential(
            nn.Conv1d(36, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

    def forward_align(self, e_t, f_t):
        # 1. 提取波形特征并对齐至 256
        e_wave = self.eeg_wave_conv(e_t)
        e_wave = F.interpolate(e_wave, size=256, mode='linear', align_corners=False)

        # 2. 提取瞬时能量包络 (平滑平方功率) 并对齐至 256
        e_power = torch.log1p(e_t ** 2)
        e_env = self.eeg_env_conv(self.eeg_env_pool(e_power))
        e_env = F.interpolate(e_env, size=256, mode='linear', align_corners=False)

        # 融合波形与能量包络
        e_feat = self.eeg_fuse(torch.cat([e_wave, e_env], dim=1))
        f_feat = self.fnirs_conv(f_t)
        return e_feat, f_feat