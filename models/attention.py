import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, out_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads

        self.out_dim = out_dim
        self.fc_q = nn.Linear(q_dim, out_dim, bias=False)
        self.fc_k = nn.Linear(k_dim, out_dim, bias=False)
        self.fc_v = nn.Linear(v_dim, out_dim, bias=False)
        self.fc_out = nn.Linear(out_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def scatter(self, x):
        return torch.cat(x.chunk(self.num_heads, -1), -3)

    def gather(self, x):
        return torch.cat(x.chunk(self.num_heads, -3), -1)

    def attend(self, q, k, v):
        q, k, v = [self.scatter(x) for x in [q, k, v]]
        A_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.out_dim)
        A = torch.softmax(A_logits, dim=-1)
        return self.gather(torch.matmul(A, v))

    def forward(self, q, k, v):
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        out = self.ln1(q + self.attend(q, k, v))
        out = self.ln2(out + F.relu(self.fc_out(out)))
        return out


class SelfAttention(MultiHeadAttention):
    def __init__(self, in_dim, out_dim, num_heads=8):
        super().__init__(in_dim, in_dim, in_dim, out_dim, num_heads)

    def forward(self, x):
        return super().forward(x, x, x)
