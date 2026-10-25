import torch
import torch.nn as nn
import torch.nn.functional as F

class MTRModule(nn.Module):
    def __init__(self, channels, seq_len, mode="downsample"):
        super().__init__()
        self.channels = channels
        self.mode = mode

        # 1. 多尺度深度可分离卷积 (kernels: 3, 5, 7, 9)
        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=k, stride=1, padding=k // 2, groups=channels),
                nn.GELU()
            ) for k in [3, 5, 7, 9]
        ])

        # 2. 因果膨胀卷积 (dilation=1, 2, 4, kernel_size=2)
        self.dilations = [1, 2, 4]
        self.causal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=2, stride=1, dilation=d),
                nn.GELU()
            ) for d in self.dilations
        ])

        self.stage1_proj = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.GroupNorm(min(4, channels), channels),
            nn.GELU()
        )

        # 3. 尺度变换阶段
        if mode == "downsample":
            self.pw_noadd1 = nn.Conv1d(channels, channels, kernel_size=1, stride=2)
            self.pw_noadd2 = nn.Conv1d(channels, channels, kernel_size=1, stride=2)
            self.pw_pad1 = nn.Conv1d(channels, channels, kernel_size=1, stride=2)
            self.pw_pad2 = nn.Conv1d(channels, channels, kernel_size=1, stride=2)
            self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1, stride=2)
        elif mode == "upsample":
            self.up_trans1 = nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2)
            self.up_trans2 = nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2)
            self.gate_conv = nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2)
        else:
            self.keep_convs = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size=1) for _ in range(4)
            ])
            self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1)

        self.stage2_proj = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(4, channels), channels),
            nn.GELU()
        )

    def forward(self, x):
        dw_out = sum([conv(x) for conv in self.dw_convs]) * 0.25

        c_out = x
        for d, conv in zip(self.dilations, self.causal_convs):
            c_out = F.pad(c_out, (d, 0))
            c_out = conv(c_out)

        s1_concat = torch.cat([dw_out, c_out, x], dim=1)
        s1_feat = self.stage1_proj(s1_concat)

        if self.mode == "downsample":
            p1 = self.pw_noadd1(s1_feat)
            p2 = self.pw_noadd2(s1_feat)
            p3 = self.pw_pad1(F.pad(s1_feat, (1, 0)))[:, :, :p1.shape[-1]]
            p4 = self.pw_pad2(F.pad(s1_feat, (1, 0)))[:, :, :p1.shape[-1]]
            ms_pw = F.gelu((p1 + p2 + p3 + p4) * 0.25)
            gate = torch.sigmoid(self.gate_conv(s1_feat))
        elif self.mode == "upsample":
            u1 = F.gelu(self.up_trans1(s1_feat))
            u2 = F.gelu(self.up_trans2(s1_feat))
            u3 = F.interpolate(s1_feat, scale_factor=2.0, mode='linear', align_corners=False)
            u4 = F.interpolate(s1_feat, scale_factor=2.0, mode='linear', align_corners=False)
            ms_pw = (u1 + u2 + u3 + u4) * 0.25
            gate = torch.sigmoid(self.gate_conv(s1_feat))
        else:
            ms_pw = F.gelu(sum([conv(s1_feat) for conv in self.keep_convs]) * 0.25)
            gate = torch.sigmoid(self.gate_conv(s1_feat))

        gated_feat = ms_pw * gate
        s2_concat = torch.cat([ms_pw, gated_feat], dim=1)
        return self.stage2_proj(s2_concat)