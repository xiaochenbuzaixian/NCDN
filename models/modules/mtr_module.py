import torch
import torch.nn as nn
import torch.nn.functional as F

class MTRModule(nn.Module):
    def __init__(self, channels, seq_len, mode="downsample"):
        """
        mode: "downsample" (长度减半), "upsample" (长度加倍), "keep" (保持长度)
        """
        super().__init__()
        self.channels = channels
        self.mode = mode
        out_len = seq_len // 2 if mode == "downsample" else (seq_len * 2 if mode == "upsample" else seq_len)

        # 1. 多尺度深度可分离卷积 (kernels: 3, 5, 7, 9, stride=1)
        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=k, stride=1, padding=k // 2, groups=channels),
                nn.GELU()
            ) for k in [3, 5, 7, 9]
        ])

        # 2. 因果膨胀卷积 (3层连续, dilation=1, 2, 4, kernel_size=2)
        self.dilations = [1, 2, 4]
        self.causal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=2, stride=1, dilation=d),
                nn.GELU()
            ) for d in self.dilations
        ])

        # 融合第一阶段特征 (Concat 3路特征 -> Linear 映射回原通道)
        self.stage1_proj = nn.Conv1d(channels * 3, channels, kernel_size=1)

        # 3. 尺度变换阶段 (下采样 / 上采样 / 保持)
        if mode == "downsample":
            # 4个点卷积: 2个无padding, 2个左侧零填充
            self.pw_noadd1 = nn.Conv1d(channels, channels, kernel_size=1, stride=2, padding=0)
            self.pw_noadd2 = nn.Conv1d(channels, channels, kernel_size=1, stride=2, padding=0)
            self.pw_pad1 = nn.Conv1d(channels, channels, kernel_size=1, stride=2, padding=0)
            self.pw_pad2 = nn.Conv1d(channels, channels, kernel_size=1, stride=2, padding=0)
            self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1, stride=2)
        elif mode == "upsample":
            # 2个转置卷积 + 2个插值
            self.up_trans1 = nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2)
            self.up_trans2 = nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2)
            self.gate_conv = nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2)
        else:
            self.keep_convs = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size=1) for _ in range(4)
            ])
            self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1)

        self.stage2_proj = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.out_linear = nn.Linear(out_len, out_len)

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape

        # 分支1: 多尺度深度卷积加和
        dw_out = sum([conv(x) for conv in self.dw_convs])

        # 分支2: 3层连续因果膨胀卷积 (左侧零填充 pad = dilation * (kernel_size - 1))
        c_out = x
        for d, conv in zip(self.dilations, self.causal_convs):
            c_out = F.pad(c_out, (d, 0))
            c_out = conv(c_out)

        # Concat 并通过线性变换
        s1_concat = torch.cat([dw_out, c_out, x], dim=1)  # [B, 3C, L]
        s1_feat = F.gelu(self.stage1_proj(s1_concat))     # [B, C, L]

        # 第二阶段: 下采样 / 上采样
        if self.mode == "downsample":
            p1 = self.pw_noadd1(s1_feat)
            p2 = self.pw_noadd2(s1_feat)
            p3 = self.pw_pad1(F.pad(s1_feat, (1, 0)))[:, :, :p1.shape[-1]]
            p4 = self.pw_pad2(F.pad(s1_feat, (1, 0)))[:, :, :p1.shape[-1]]
            ms_pw = F.gelu(p1 + p2 + p3 + p4)
            gate = torch.sigmoid(self.gate_conv(s1_feat))
        elif self.mode == "upsample":
            u1 = F.gelu(self.up_trans1(s1_feat))
            u2 = F.gelu(self.up_trans2(s1_feat))
            u3 = F.interpolate(s1_feat, scale_factor=2.0, mode='linear', align_corners=False)
            u4 = F.interpolate(s1_feat, scale_factor=2.0, mode='linear', align_corners=False)
            ms_pw = u1 + u2 + u3 + u4
            gate = torch.sigmoid(self.gate_conv(s1_feat))
        else:
            ms_pw = F.gelu(sum([conv(s1_feat) for conv in self.keep_convs]))
            gate = torch.sigmoid(self.gate_conv(s1_feat))

        gated_feat = ms_pw * gate
        s2_concat = torch.cat([ms_pw, gated_feat], dim=1)
        out = self.stage2_proj(s2_concat)
        return self.out_linear(out)