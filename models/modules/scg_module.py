import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCGModule(nn.Module):
    def __init__(self, d_in, d_out, topo_in_ch=30, topo_out_ch=36):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.conv2d_q = nn.Conv2d(topo_in_ch, d_in, kernel_size=3, padding=1)
        self.conv2d_k = nn.Conv2d(topo_out_ch, d_out, kernel_size=3, padding=1)

        self.gate_conv = nn.Conv1d(d_in, d_out, kernel_size=1)
        self.res_proj = nn.Conv1d(d_in, d_out, kernel_size=1)
        self.norm = nn.GroupNorm(min(4, d_out), d_out)

    def forward(self, v_feat, topo_q, topo_k):
        B, _, L = v_feat.shape

        Q = self.conv2d_q(topo_q.unsqueeze(0)).view(1, self.d_in, -1)
        K = self.conv2d_k(topo_k.unsqueeze(0)).view(1, self.d_out, -1)

        attn_logits = torch.bmm(K, Q.transpose(1, 2)) / math.sqrt(self.d_out)
        score = F.softmax(attn_logits, dim=-1).expand(B, -1, -1)

        attn_out = torch.bmm(score, v_feat)
        gate = torch.sigmoid(self.gate_conv(v_feat))

        # 引入残差直连与归一化，防止多层 Sigmoid 导致梯度消失
        out = self.norm(attn_out * gate + self.res_proj(v_feat))
        return F.gelu(out)