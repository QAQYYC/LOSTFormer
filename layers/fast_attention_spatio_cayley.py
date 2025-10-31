import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
from layers.fast_attention_utils import *


class PerformerAttention(nn.Module):
    def __init__(self, config, nb_features=None, use_relu_kernel=True, seed=0, d_model=None):
        super().__init__()
        self.device = config.device
        self.heads = config.n_heads
        self.d_model = d_model if d_model is not None else config.d_model
        self.d_head = self.d_model // self.heads
        if nb_features is None:
            self.nb_features = self.d_head
        else:
            self.nb_features = nb_features
        self.use_relu_kernel = use_relu_kernel

        self.to_qkv = nn.Linear(self.d_model, self.d_model * 3, bias=False)
        self.to_out = nn.Linear(self.d_model, self.d_model)

        projection = create_projection_matrix(self.nb_features, self.d_head, seed=seed, scaling=1)
        self.projection_adjust = nn.Parameter(create_projection_matrix(self.d_head, self.d_head, seed=seed, scaling=1).transpose(-1, -2), requires_grad=True)
        self.register_buffer('projection', projection)
        self.scaling_matrix = torch.diag(math.sqrt(self.projection.shape[1]) * torch.ones((self.projection.shape[0]))).to(self.device)

    def forward(self, x, _, __, cache=None, masked=False):
        B, C, P, D = x.shape
        H, D_H = self.heads, self.d_head

        x = x.permute(0,2,1,3).reshape(B, C * P, D)  # [B, C*P, D]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B , C * P, H, D_H) for t in qkv]
        del qkv

        projection_adj = self.projection_adjust - self.projection_adjust.t()
        projection_adj = torch.inverse(torch.eye(self.d_head, device=self.device) - projection_adj) @ (torch.eye(self.d_head, device=self.device) + projection_adj)
        projection_matrix = self.scaling_matrix @ self.projection @ projection_adj  # [m, d] @ [d, d] -> [m, d]
        del projection_adj
        # Optional caching for inference
        if cache is not None:
            if 'k' in cache and 'v' in cache:
                k = torch.cat([cache['k'], k], dim=1)
                v = torch.cat([cache['v'], v], dim=1)
            cache['k'], cache['v'] = k, v

        kernel_fn = relu_kernel_transformation if self.use_relu_kernel else softmax_kernel_transformation
        q_prime = kernel_fn(q, projection_matrix) if self.use_relu_kernel else kernel_fn(q, True, projection_matrix)
        k_prime = kernel_fn(k, projection_matrix) if self.use_relu_kernel else kernel_fn(k, False, projection_matrix)
        del q
        del k

        # Permute to [S, B, H, *]
        q_prime = q_prime.permute(1, 0, 2, 3)
        k_prime = k_prime.permute(1, 0, 2, 3)
        v = v.permute(1, 0, 2, 3)
        if masked:
            num = causal_numerator(q_prime, k_prime, v, channel=C)
            den = causal_denominator(q_prime, k_prime, channel=C)
        else:
            kvs = torch.einsum("sbhm,sbhd->bhmd", k_prime, v)
            ks_sum = torch.sum(k_prime, dim=0)
            del v
            num = torch.einsum("sbhm,bhmd->sbhd", q_prime, kvs)
            den = torch.einsum("sbhm,bhm->sbh", q_prime, ks_sum)
            del kvs
            del ks_sum

        out = num / den.unsqueeze(-1)
        del num
        del den
        out = out.transpose(0,1).reshape(B, P * C, H * D_H)
        out = self.to_out(out).view(B, P, C, D).permute(0, 2, 1, 3)  # [B,C,P,D]

        return out, (q_prime, k_prime, projection_matrix)
