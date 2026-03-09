import os
import sys
import math
import torch
import random
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from mask_layer import TransformerEncoder, TransformerEncoderLayer
from positional_encodings.torch_encodings import PositionalEncoding2D
from moeExpert import SoftMoE


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
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1,
                 alph=0.3, gama=0.8, lambda_sparse=1e-3, lambda_entropy=1e-3):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.alph = alph
        self.gama = gama

        # 正则系数放在模块里，方便在 config 调
        self.lambda_sparse = lambda_sparse
        self.lambda_entropy = lambda_entropy

        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.agsc = AGSG(num_nodes, channels, alph)
        self.conv_gcn = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_reg=True):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()  # -> (B, C, N, T)
        skip = x

        x = self.conv(x)
        A_p = self.agsc(x)   
        A_f0 = torch.softmax(A_p, dim=-1)   

        k = int(A_f0.shape[1] * self.gama)
        _, topk_idx = torch.topk(A_f0, k=k, dim=-1)
        mask = torch.zeros_like(A_f0)
        mask.scatter_(-1, topk_idx, 1.0)

        # 用 mask 得到稀疏图，并重新归一化，避免行和 < 1 影响传播稳定性
        A_f = A_f0 * mask

        # ===== 正则项：放在这里（用 A_f0 或 A_f 都行，但语义不同） =====
        # (1) sparse：惩罚 topk 之外的“尾部概率质量”（更像在做无监督结构发现）
        tail_mass = (A_f0 * (1.0 - mask)).sum(dim=-1).mean()  # 标量
        loss_graph = self.lambda_sparse * tail_mass

        # (2) entropy：鼓励每行分布更“尖”（更稀疏、更确定）
        entropy = -(A_f0 * torch.log(A_f0 + 1e-8)).sum(dim=-1).mean()
        loss_entropy = self.lambda_entropy * entropy

        reg_loss = loss_graph + loss_entropy
        # ============================================================

        out = []
        x_prop = x
        for _ in range(self.diffusion_step):
            x_prop = torch.einsum("bcnt,bnm->bcmt", x_prop, A_f).contiguous()
            out.append(x_prop)

        x = torch.cat(out, dim=1)
        x = self.conv_gcn(x)
        x = self.dropout(x)
        x = x + skip
        x = x.transpose(1, 2).transpose(2, 3).contiguous()

        if return_reg:
            return x, reg_loss
        return x    
    
class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, num_nodes=307, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, num_nodes, hidden_dim*mlp_ratio, dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.moe = SoftMoE(hidden_dim, 3, 10)
        self.dggc =DGGC(channels=hidden_dim, num_nodes=num_nodes, diffusion_step=1, dropout=dropout, gama=0.8)
        
    def forward(self, src):
        
        B, N, L, D = src.shape
        
        src = src * math.sqrt(self.d_model)
        src, reg_loss = self.dggc(src)
        src=src.contiguous()
        src = src.view(B*N, L, D).contiguous()
        src = src.transpose(0, 1).contiguous()

        output = self.transformer_encoder(src)
        output = output + self.moe(output)
        output = output.transpose(0, 1).view(B, N, L, D).contiguous()
        return output, reg_loss

class PositionalEncoding(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, input_data, index=None, abs_idx=None):

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        tp_enc_2d = PositionalEncoding2D(num_feat).to(input_data.device)
        input_data+=tp_enc_2d(input_data).to(input_data.device)
        
        return input_data,tp_enc_2d(input_data)
    
    
class MaskGenerator(nn.Module):

    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens
    

class PatchEmbedding(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
                                        in_channel,
                                        embed_dim,
                                        kernel_size=(self.len_patch, 1),
                                        stride=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):

        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1) # B, N, C, L, 1
        # B*N,  C, L, 1
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1).contiguous()

        # B*N,  d, L/P, 1
        output = self.input_embedding(long_term_history)
        # norm
        output = self.norm_layer(output)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1).contiguous()    # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch
        return output

def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class Mask(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, num_nodes, num_heads, mlp_ratio, dropout,  mask_ratio, encoder_depth, decoder_depth, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = 0
        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat=None
        # encoder specifics
        # # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding()

        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, num_nodes, dropout)

        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.xavier_uniform_(self.mask_token)
        # # decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, num_nodes, dropout)

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):

        # patchify and embed input
        if mask:
            # mask: True
            patches = self.patch_embedding(long_term_history)  # B, N, d, P
            # patches: torch.Size([16, 307, 96, 72])
            patches = patches.transpose(-1, -2).contiguous()  # B, N, P, d
            batch_size, num_nodes, num_time,num_dim  =  patches.shape

            # positional embedding
            patches,self.pos_mat = self.positional_encoding(patches)        # mask
            Maskg=MaskGenerator(patches.shape[2], self.mask_ratio)
            unmasked_token_index, masked_token_index = Maskg.uniform_rand()
            encoder_input = patches[:, :, unmasked_token_index, :]
            hidden_states_unmasked, reg_loss = self.encoder(encoder_input)
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim).contiguous()
            return hidden_states_unmasked, unmasked_token_index, masked_token_index, reg_loss

        else:
            batch_size, num_nodes, _, _ = long_term_history.shape  # long_term_history: torch.Size([32, 170, 1, 1152])
            # patchify and embed input
            patches = self.patch_embedding(long_term_history)     # B, N, d, P
            # patches: torch.Size([32, 170, 96, 96])
            patches = patches.transpose(-1, -2).contiguous()         # B, N, P, d
            # patches: torch.Size([32, 170, 96, 96])
            # positional embedding
            patches,self.pos_mat = self.positional_encoding(patches)# B, N, P, d
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches # B, N, P, d
            hidden_states_unmasked, reg_loss = self.encoder(encoder_input)# B,  P,N, d/# B, N, P, d
            # hidden_states_unmasked: torch.Size([32, 170, 96, 96])
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim).contiguous() # B, N, P, d

            return hidden_states_unmasked, unmasked_token_index, masked_token_index, reg_loss

    def decoding(self, hidden_states_unmasked, masked_token_index):

        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked) # B, N, P, d/# B,P, N,  d
        # B,N*r,P,d
        batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
        unmasked_token_index=[i for i in range(0,len(masked_token_index)+num_time) if i not in masked_token_index ]
        hidden_states_masked = self.pos_mat[:,:,masked_token_index,:]
        hidden_states_masked+=self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1])
        hidden_states_unmasked+=self.pos_mat[:,:,unmasked_token_index,:]
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d

        # decoding
        hidden_states_full, reg_loss = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)

        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim).contiguous())

        return reconstruction_full, reg_loss

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index,
                                        masked_token_index):
        # get reconstructed masked tokens
        batch_size, num_nodes, num_time, _ = reconstruction_full.shape

        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2).contiguous()     # B, r*P*d, N

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2).contiguous()  # B, N, P, L
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2).contiguous()  # B, r*P*d, N

        return reconstruction_masked_tokens, label_masked_tokens

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1).contiguous()     # B, N, 1, L * P
        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index, reg_loss_a = self.encoding(history_data)
            # decoding
            reconstruction_full, reg_loss_b = self.decoding(hidden_states_unmasked, masked_token_index)
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)
            return reconstruction_masked_tokens, label_masked_tokens, reg_loss_a+reg_loss_b
        else:
            hidden_states_full, _, _,_ = self.encoding(history_data, mask=False)
            return hidden_states_full


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    model = Mask(
            patch_size=12,
            in_channel=1,
            embed_dim=96,
            num_nodes=307,
            num_heads=4,
            mlp_ratio=4,
            dropout=0.1,
            mask_ratio=0.25,
            encoder_depth=4,
            decoder_depth=1,
            mode="pre-train"
        )
    x = torch.empty(24, 864, 307, 1)
    print("x:", x.shape)
    out_x, out_y = model(x)
    print("out_x:{}, out_y:{}".format(out_x.shape, out_y.shape))
