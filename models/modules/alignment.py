import torch
import torch.nn as nn

class InputAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        # EEG: [B, 30, 4000] -> [B, 32, 256]
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(30, 32, kernel_size=7, stride=5, padding=3),  # 4000 -> 800
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=3, padding=2),  # 800 -> 267
            nn.GELU()
        )
        self.eeg_linear = nn.Linear(267, 256)

        # fNIRS: [B, 36, 256] -> [B, 32, 256]
        self.fnirs_conv = nn.Conv1d(36, 32, kernel_size=1)
        self.fnirs_linear = nn.Linear(256, 256)

        # 逆向输出还原层: [B, 32, 256] -> [B, 36, 256]
        self.out_proj = nn.Conv1d(32, 36, kernel_size=1)

    def forward_align(self, e_t, f_t):
        e_feat = self.eeg_linear(self.eeg_conv(e_t))
        f_feat = self.fnirs_linear(self.fnirs_conv(f_t))
        return e_feat, f_feat

    def forward_restore(self, f_latent):
        return self.out_proj(f_latent)