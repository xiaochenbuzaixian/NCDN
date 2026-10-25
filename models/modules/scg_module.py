import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCGModule(nn.Module):
    def __init__(self, d_in, d_out, topo_in_ch=30, topo_out_ch=36):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # 对 16x16 相关性平面执行 2D 卷积生成 Q 和 K
        self.conv2d_q = nn.Conv2d(topo_in_ch, d_in, kernel_size=3, padding=1)
        self.conv2d_k = nn.Conv2d(topo_out_ch, d_out, kernel_size=3, padding=1)

        # 门控卷积分支 (对应 Fig. 1 左下角 Conv1D + Sigmoid)
        self.gate_conv = nn.Conv1d(d_in, d_out, kernel_size=1)
        self.out_gate = nn.Conv1d(d_out, d_out, kernel_size=1)

    def forward(self, v_feat, topo_q, topo_k):
        """
        v_feat: [B, d_in, L] 输入表征 V
        topo_q: [topo_in_ch, 16, 16]
        topo_k: [topo_out_ch, 16, 16]
        返回: [B, d_out, L]
        """
        B, _, L = v_feat.shape

        # 计算 Q: [1, d_in, 256], K: [1, d_out, 256]
        Q = self.conv2d_q(topo_q.unsqueeze(0)).view(1, self.d_in, -1)
        K = self.conv2d_k(topo_k.unsqueeze(0)).view(1, self.d_out, -1)

        # 计算注意力矩阵 Score: [1, d_out, d_in]
        attn_logits = torch.bmm(K, Q.transpose(1, 2)) / math.sqrt(self.d_out)
        score = F.softmax(attn_logits, dim=-1).expand(B, -1, -1)  # [B, d_out, d_in]

        # 映射表征: Score * V -> [B, d_out, L]
        attn_out = torch.bmm(score, v_feat)

        # 门控融合 (Fig. 1 SCG 底部结构)
        gate_in = torch.sigmoid(self.gate_conv(v_feat))
        combined = attn_out * gate_in
        out = torch.sigmoid(self.out_gate(combined)) * combined
        return out