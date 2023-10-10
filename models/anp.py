import math

import torch
import torch.nn as nn
from collections import defaultdict

from models.building_blocks import DetermininsticAttentionEncoder, LatentEncoder, Decoder
from utils.misc import stack_tensor


class ANP(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, r_dim=128, z_dim=128, h_dim=128,
                       enc_pre_depth=4, enc_post_depth=2,
                       qk_depth=2, v_depth=4, self_attn=False,
                       dec_depth=3):
        super().__init__()

        self.encoder1 = DetermininsticAttentionEncoder(x_dim, y_dim, r_dim, h_dim,
                                                       qk_depth=qk_depth,
                                                       v_depth=v_depth,
                                                       self_attn=self_attn)

        self.encoder2 = LatentEncoder(x_dim, y_dim, z_dim, h_dim,
                                     pre_depth=enc_pre_depth,
                                     post_depth=enc_post_depth,
                                     self_attn=self_attn)

        self.decoder = Decoder(x_dim, y_dim, r_dim + z_dim, h_dim,
                               depth=dec_depth)

        # for BoTorch
        self.num_outputs = 1
        self.x_init = None
        self.y_init = None

    def forward(self, x_context, y_context, x_target, y_target=None, num_samples=None):
        if self.training:
            num_target = x_target.shape[-2]
            r_target = self.encoder1(x_context, y_context, x_target)

            q_context = self.encoder2(x_context, y_context)
            q_target = self.encoder2(x_target, y_target)

            if num_samples is not None:
                r_target = stack_tensor(r_target, num_samples, dim=0)
                z = q_target.rsample([num_samples])
            else:
                z = q_target.rsample()

            enc = torch.cat([r_target, stack_tensor(z, num_target, dim=-2)], dim=-1)

            if num_samples is not None:
                x_target = stack_tensor(x_target, num_samples, dim=0)
            p_y_pred = self.decoder(x_target, enc)
            return p_y_pred, q_context, q_target, z
        else:
            num_target = x_target.shape[-2]
            r_target = self.encoder1(x_context, y_context, x_target)

            q_context = self.encoder2(x_context, y_context)
            if num_samples is not None:
                r_target = stack_tensor(r_target, num_samples, dim=0)
                z = q_context.rsample([num_samples])
            else:
                z = q_context.rsample()
            
            enc = torch.cat([r_target, stack_tensor(z, num_target, dim=-2)], dim=-1)
            
            if num_samples is not None:
                x_target = stack_tensor(x_target, num_samples, dim=0)
            p_y_pred = self.decoder(x_target, enc)
            return p_y_pred
        
    def calculate_loss(self, out, y, num_samples):
        p_y_pred, q_context, q_target, z = out
        
        y = stack_tensor(y, num_samples, dim=0)
        ll = p_y_pred.log_prob(y).sum(dim=-1)

        log_qz = q_target.log_prob(z).sum(-1)
        log_pz = q_context.log_prob(z).sum(-1)

        log_w = ll.sum(-1) + log_pz - log_qz
        loss = -torch.logsumexp(log_w, dim=0) + math.log(num_samples)
        return loss.mean() / y.shape[-2]

    def evaluate(self, p_y_pred, y_context, y_target, num_samples):
        result = defaultdict(list)
        num_context = y_context.shape[-2]

        y_target = stack_tensor(y_target, num_samples, dim=0)
        ll = p_y_pred.log_prob(y_target).sum(dim=-1)
        result["context_ll"] = ll[..., :num_context].mean().item()
        result["target_ll"] = ll[..., num_context:].mean().item()
        return result

    def posterior(self, X, posterior_transform=None):
        num_target = X.shape[-2]
        x_context = stack_tensor(self.x_init, X.shape[0], dim=0)
        y_context = stack_tensor(self.y_init, X.shape[0], dim=0)

        r_target = self.encoder1(x_context, y_context, X)
        q_context = self.encoder2(x_context, y_context)
        z = q_context.rsample()

        enc = torch.cat([r_target, stack_tensor(z, num_target, dim=-2)], dim=-1)
        
        p_y_pred = self.decoder(X, enc)
        return p_y_pred
