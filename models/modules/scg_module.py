import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCGModule(nn.Module):
    """
    原论文 Spatial Cross-modal Generation (SCG) 空间跨模态注意力模块:
    结合 16x16 空间相关拓扑先验与当前试次动态特征，实现 d_in -> d_out 的跨模态空间路由
    """
    def __init__(self, d_in, d_out, topo_in_ch=30, topo_out_ch=36):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.conv2d_q = nn.Sequential(
            nn.Conv2d(topo_in_ch, d_in, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(d_in, d_in, kernel_size=3, padding=1)
        )
        self.conv2d_k = nn.Sequential(
            nn.Conv2d(topo_out_ch, d_out, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(d_out, d_out, kernel_size=3, padding=1)
        )
        # 结合试次动态上下文调节空间注意力
        self.dyn_q = nn.Linear(d_in, d_in)
        self.gate_conv = nn.Conv1d(d_in, d_out, kernel_size=1)
        self.res_proj = nn.Conv1d(d_in, d_out, kernel_size=1)
        num_groups = min(4, d_out) if d_out % min(4, d_out) == 0 else 1
        self.norm = nn.GroupNorm(num_groups, d_out)

    def forward(self, v_feat, topo_q, topo_k):
        B, _, _ = v_feat.shape
        tq = topo_q.unsqueeze(0) if topo_q.dim() == 3 else topo_q[:1]
        tk = topo_k.unsqueeze(0) if topo_k.dim() == 3 else topo_k[:1]

        Q_static = self.conv2d_q(tq).flatten(2).expand(B, -1, -1)  # (B, d_in, 256)
        K_static = self.conv2d_k(tk).flatten(2).expand(B, -1, -1)  # (B, d_out, 256)

        # 注入当前试次通道能量上下文，使空间路由随试次动态调整
        v_ctx = self.dyn_q(v_feat.mean(dim=-1)).unsqueeze(-1)       # (B, d_in, 1)
        Q = Q_static * (1.0 + torch.tanh(v_ctx))

        attn_logits = torch.bmm(K_static, Q.transpose(1, 2)) / math.sqrt(256.0)
        score = F.softmax(attn_logits, dim=-1)                      # (B, d_out, d_in)

        attn_out = torch.bmm(score, v_feat)
        gate = torch.sigmoid(self.gate_conv(v_feat))
        out = self.norm(attn_out * gate + self.res_proj(v_feat))
        return F.gelu(out)