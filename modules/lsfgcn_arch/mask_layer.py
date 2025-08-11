import os
import sys
import copy
from typing import Optional, Any, Union, Callable, Tuple, Literal
from torch import nn
import torch
import numpy as np
from torch import Tensor
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import math
from einops import rearrange
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
  
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    def forward(self, src: Tensor) -> Tensor:
       
        for idx, mod in enumerate(self.layers):
            src = mod(src)
        
        return src


class AGSG(nn.Module):
    def __init__(self, num_nodes, channels, alph):
        super(AGSG, self).__init__()
        self.alph = alph
        self.num_nodes = num_nodes
        self.channels = channels
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)

    def forward(self, x):
        initial_S = F.relu(torch.mm(self.memory.transpose(0, 1).contiguous(), self.memory)).to(x.device)
        initial_S = torch.where(torch.eye(self.num_nodes, device=x.device) == 1, torch.full_like(initial_S, 0.1), initial_S)

        S_w = F.softmax(initial_S, dim=1).to(x.device)
        support_set = [torch.eye(self.num_nodes).to(x.device), S_w]

        for k in range(2, self.num_nodes + 1):
            support_set.append(torch.mm(S_w, support_set[k - 1]))

        supports = torch.stack(support_set, dim=0).to(x.device)
        
        A_p = torch.softmax(F.relu(torch.einsum("bcnt, knm->bnm", x, supports).contiguous() / math.sqrt(x.shape[1])), -1)
        return A_p


class DGGC(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, alph=0.3, gama=0.8):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.alph = alph
        self.gama = gama
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.agsc = AGSG(num_nodes, channels, alph)
        self.fc = nn.Conv2d(2, 1, (1, 1))
        self.conv_gcn = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        x = self.conv(x)
        A_p = self.agsc(x)

        A_f = torch.softmax(A_p, -1)

        topk_values, topk_indices = torch.topk(A_f, k=int(A_f.shape[1] * self.gama), dim=-1)
        mask = torch.zeros_like(A_f)
        mask.scatter_(-1, topk_indices, 1)
        A_f = A_f * mask

        out = []
        for i in range(self.diffusion_step):
            x = torch.einsum("bcnt,bnm->bcmt", x, A_f).contiguous()
            out.append(x)

        x = torch.cat(out, dim=1)
        x = self.conv_gcn(x)
        x = self.dropout(x)
        x = x.transpose(1, 2).transpose(2, 3).contiguous()
        return x

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']
    
    # hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout
    def __init__(self, d_model: int, nhead: int, num_nodes: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        
        # self.dggc =DGGC(channels=d_model, num_nodes=num_nodes, diffusion_step=1, dropout=dropout, gama=0.8)
        # self.moe = SoftMoE(d_model, 3, 64)
        # self.moe_norm = Norm(d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        self.activation_relu_or_gelu = 1
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self._sa_block(x))
        # x = self.norm3(x + self.moe(x))
        x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class NonDynamicallyQuantizableLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)

class Norm(nn.Module):
    def __init__(self, embed_dim, include_bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if include_bias else None

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# class SoftMoe(nn.Module):
#     def __init__(self, embed_dim, n_experts, slots_per_expert, dropout):
#         super().__init__()
#         self.slots_per_expert = slots_per_expert
#         self.experts = nn.ModuleList([FeedForward(embed_dim, dropout) for _ in range(n_experts)])
#         self.phi = nn.Parameter(torch.randn(embed_dim, n_experts * slots_per_expert))
        
#     def forward(self, x: torch.Tensor):
#         logits = torch.matmul(x, self.phi) # (batch_size, seq_len, slots)
#         dispatch_weights = F.softmax(logits, dim=-1)
#         combine_weights = F.softmax(logits, dim=1)
#         xs = torch.bmm(dispatch_weights.transpose(1, 2), x)
#         ys = torch.cat(
#             [expert(xs[:, i * self.slots_per_expert : (i + 1) * self.slots_per_expert, :]) 
#                           for i, expert in enumerate(self.experts)],
#             dim=1
#             )
#         y = torch.bmm(combine_weights, ys)
#         return y
    

class MultiheadAttention(nn.Module):
   
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        
        
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
       
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
    
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Optional[Tensor]]:

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False)
        
        return attn_output, attn_output_weights