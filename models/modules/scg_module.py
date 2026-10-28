import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCGModule(nn.Module):
    """原论文 Spatial Cross-modal Generation (SCG) 空间跨模态注意力模块"""
    def __init__(self, d_in, d_out, topo_in_ch=30, topo_out_ch=36):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.conv2d_q = nn.Conv2d(topo_in_ch, d_in, kernel_size=3, padding=1)
        self.conv2d_k = nn.Conv2d(topo_out_ch, d_out, kernel_size=3, padding=1)
        self.gate_conv = nn.Conv1d(d_in, d_out, kernel_size=1)
        self.res_proj = nn.Conv1d(d_in, d_out, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, d_out), d_out)

    def forward(self, v_feat, topo_q, topo_k):
        B, _, _ = v_feat.shape
        tq = topo_q.unsqueeze(0) if topo_q.dim() == 3 else topo_q[:1]
        tk = topo_k.unsqueeze(0) if topo_k.dim() == 3 else topo_k[:1]
        Q = self.conv2d_q(tq).flatten(2)  # (1, d_in, 256)
        K = self.conv2d_k(tk).flatten(2)  # (1, d_out, 256)
        attn_logits = torch.bmm(K, Q.transpose(1, 2)) / math.sqrt(256.0)
        score = F.softmax(attn_logits, dim=-1).expand(B, -1, -1)  # (B, d_out, d_in)
        attn_out = torch.bmm(score, v_feat)
        gate = torch.sigmoid(self.gate_conv(v_feat))
        out = self.norm(attn_out * gate + self.res_proj(v_feat))
        return F.gelu(out)