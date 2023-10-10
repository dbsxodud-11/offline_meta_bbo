
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from models.attention import MultiHeadAttention, SelfAttention


class BaseMLP(nn.Module):
    def __init__(self, input_dim, output_dim,
                       hidden_dim=64, depth=3):
        super(BaseMLP, self).__init__()

        input_dims = [input_dim] + [hidden_dim] * (depth-1)
        output_dims = [hidden_dim] * (depth - 1) + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            layer = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)  

            if i < depth-1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, r_dim=128, h_dim=128,
                       pre_depth=4, post_depth=2):
        super().__init__()

        self.pre_model = BaseMLP(x_dim+y_dim, h_dim, h_dim, pre_depth)
        self.post_model = BaseMLP(h_dim, r_dim, h_dim, post_depth)

    def forward(self, x, y):
        out = self.pre_model(torch.cat([x, y], dim=-1))
        out = out.mean(dim=-2)
        return self.post_model(out)

class DetermininsticAttentionEncoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, r_dim=128, h_dim=128,
                       qk_depth=2, v_depth=4, self_attn=False):
        super().__init__()

        if not self_attn:
            self.v_model = BaseMLP(x_dim+y_dim, h_dim, h_dim, v_depth)
        else:
            self.v_model = nn.Sequential(BaseMLP(x_dim+y_dim, h_dim, h_dim, v_depth-2),
                                         nn.ReLU(),
                                         SelfAttention(h_dim, h_dim))

        self.qk_model = BaseMLP(x_dim, r_dim, h_dim, qk_depth)
        self.cross_attention = MultiHeadAttention(h_dim, h_dim, h_dim, r_dim)

    def forward(self, x_context, y_context, x_target):
        q, k = self.qk_model(x_target), self.qk_model(x_context)
        v = self.v_model(torch.cat([x_context, y_context], dim=-1))

        return self.cross_attention(q, k, v)
    

class LatentEncoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, z_dim=128, h_dim=128,
                       pre_depth=4, post_depth=2, self_attn=False):
        super().__init__()

        if not self_attn:
            self.pre_model = BaseMLP(x_dim+y_dim, h_dim, h_dim, pre_depth)
        else:
            self.pre_model = nn.Sequential(BaseMLP(x_dim+y_dim, h_dim, h_dim, pre_depth-2),
                                           nn.ReLU(),
                                           SelfAttention(h_dim, h_dim))
                                           
        self.post_model = BaseMLP(h_dim, z_dim*2, h_dim, post_depth)

    def forward(self, x, y):
        out = self.pre_model(torch.cat([x, y], dim=-1))
        out = out.mean(dim=-2)

        mu, pre_sigma = self.post_model(out).chunk(2, dim=-1)
        sigma = 0.1 + 0.9 * torch.sigmoid(pre_sigma)
        return Normal(mu, sigma)


class Decoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, enc_dim=128, h_dim=128,
                       depth=3):
        super().__init__()

        self.model = BaseMLP(x_dim + enc_dim, y_dim*2, h_dim, depth)
    
    def forward(self, x, enc):
        mu, pre_sigma = self.model(torch.cat([x, enc], dim=-1)).chunk(2, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)
