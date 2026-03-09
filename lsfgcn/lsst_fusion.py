import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.linear1 = nn.Linear(features, features)
        self.linear2 = nn.Linear(features, features)
        self.linear3 = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.linear3(out)
        return out.permute(0, 3, 2, 1)


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.time_day = nn.Embedding(time, int(features / 2))
        self.time_week = nn.Embedding(7, int(features / 2))

    def forward(self, x):
        features = []
        day_emb = x[..., 1]

        time_day = self.time_day(
            (day_emb * self.time).long()
        )

        time_day = time_day.transpose(1, 2).contiguous()

        week_emb = x[..., 2]

        time_week = self.time_week(
            week_emb.long()
        )

        time_week = time_week.transpose(1, 2).contiguous()

        features.append(time_day)
        features.append(time_week)

        tem_emb = torch.cat(features, dim=-1)
        tem_emb = tem_emb.permute(0, 3, 1, 2)


        return tem_emb


class AGSG(nn.Module):
    def __init__(self, num_nodes, channels, alph):
        super(AGSG, self).__init__()
        self.alph = alph
        self.num_nodes = num_nodes
        self.channels = channels
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)

    def forward(self, x):
        
        initial_S = F.relu(torch.mm(self.memory.transpose(0, 1), self.memory)).to(x.device)
        initial_S = torch.where(torch.eye(self.num_nodes, device=x.device) == 1, torch.full_like(initial_S, 0.1), initial_S)
        S_w = F.softmax(initial_S, dim=1).to(x.device)
        support_set = [torch.eye(self.num_nodes).to(x.device), S_w]

        for k in range(2, self.num_nodes + 1):
            support_set.append(torch.mm(S_w, support_set[k - 1]))

        supports = torch.stack(support_set, dim=0).to(x.device)
        A_p = torch.softmax(F.relu(torch.einsum("bcnt, knm->bnm", x, supports).contiguous() / math.sqrt(x.shape[1])), -1)
        return A_p


class DGGC(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, alph=0.0, gama=0.8, emb=None, lambda_sparse=1e-3, lambda_entropy=1e-3):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.alph = alph
        self.gama = gama
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.agsc = AGSG(num_nodes, channels, alph)
        self.lambda_sparse = lambda_sparse
        self.lambda_entropy = lambda_entropy
        self.emb = emb
        self.conv_gcn = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        skip = x
        x = self.conv(x)
        A_p = self.agsc(x)
        A_f = torch.softmax(A_p, -1)
        
        
        topk_values, topk_indices = torch.topk(A_f, k=int(A_f.shape[1] * self.gama), dim=-1)
        mask = torch.zeros_like(A_f)
        mask.scatter_(-1, topk_indices, 1)
        A_f1 = A_f * mask

        # ===== 正则项：放在这里（用 A_f0 或 A_f 都行，但语义不同） =====
        # (1) sparse：惩罚 topk 之外的“尾部概率质量”（更像在做无监督结构发现）
        tail_mass = (A_f * (1.0 - mask)).sum(dim=-1).mean()  # 标量
        loss_graph = self.lambda_sparse * tail_mass

        # (2) entropy：鼓励每行分布更“尖”（更稀疏、更确定）
        entropy = -(A_f * torch.log(A_f + 1e-8)).sum(dim=-1).mean()
        loss_entropy = self.lambda_entropy * entropy

        reg_loss = loss_graph + loss_entropy
        # ============================================================

        out = []
        for i in range(self.diffusion_step):
            x = torch.einsum("bcnt,bnm->bcmt", x, A_f1).contiguous()
            out.append(x)

        x = torch.cat(out, dim=1)
        x = self.conv_gcn(x)
        x = self.dropout(x)
        x = x * self.emb + skip
        return x, reg_loss


class STIF(nn.Module):
    def __init__(self, alph, gama, channels=64, diffusion_step=1, num_nodes=170, dropout=0.2, emb=None):
        super(STIF, self).__init__()
        self.dropout = dropout
        self.num_nodes = num_nodes

        pad_l, pad_r = 3, 3
        k1, k2 = 5, 3

        self.tconv1 = nn.Sequential(
                nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
                nn.Conv2d(channels, channels, kernel_size=(1, k1)),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv2d(channels, channels, kernel_size=(1, k2)),
                nn.Tanh(),
            )

        self.dggc = DGGC(channels, num_nodes, diffusion_step, dropout, alph, gama, emb)

    def forward(self, x):
        x1 = self.tconv1(x)
        x1, reg_loss = self.dggc(x1)
        return x1, reg_loss

class STGIF(nn.Module):
    def __init__(self, history_seq_len, channels=64, diffusion_step=1, num_nodes=170, dropout=0.1, alph=0.0, gama=0.8):
        super(STGIF, self).__init__()

        self.library1 = nn.Parameter(torch.randn(channels, num_nodes, history_seq_len))

        self.STIF1 = STIF(alph, gama, channels=channels, diffusion_step=diffusion_step, num_nodes=num_nodes, dropout=dropout, emb=self.library1)

    def forward(self, x):
        concat0, reg_loss = self.STIF1(x)
        output = concat0 + x
        return output, reg_loss

class STGNet(nn.Module):
    def __init__(self, history_seq_len, future_seq_len, input_dim, num_nodes, channels, granularity, long_term_history, dropout=0.1, alph=0.0, gama=0.8):
        super(STGNet, self).__init__()
        self.num_nodes = num_nodes
        self.output_len = future_seq_len
        self.long_term_history = long_term_history
        self.history_seq_len = history_seq_len

        self.fc_his_t =  nn.Sequential(
                nn.ReplicationPad2d((3, 3, 0, 0)),
                nn.Conv2d(96, 258, kernel_size=(1, 5)),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(258,channels*2, kernel_size=(1, 3)),
                nn.Tanh(),
            )
        
        self.conv_layer = nn.Sequential(
                nn.ReplicationPad2d((3, 3, 0, 0)),
                nn.Conv2d(channels, channels, kernel_size=(1, 5)),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(channels, future_seq_len, kernel_size=(1, 3)),
                nn.Tanh(),
            )
        self.Temb = TemporalEmbedding(granularity, channels)
        self.input_proj = nn.Linear(input_dim, channels)

        self.tree = STGIF(history_seq_len=history_seq_len, channels=channels * 2, diffusion_step=2, num_nodes=self.num_nodes, dropout=dropout, alph=alph, gama=gama)
        self.glu = GLU(channels * 2, dropout)
        self.regression_layer = nn.Conv2d(channels * 2, self.output_len, kernel_size=(1, self.output_len))

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input, hidden_states):
        
        input = input.transpose(1, 3)
        
        hidden_states_new = self.conv_layer(hidden_states.transpose(1, 2)).transpose(1, 3)
        hidden_states_t = self.fc_his_t(hidden_states_new) 
        
        x = input
        time_emb = self.Temb(input.permute(0, 3, 2, 1))
        
        x = torch.cat([self.input_proj(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1), time_emb], dim=1)
        x, reg_loss = self.tree(x)
        x = x + hidden_states_t
        
        glu = self.glu(x) + x
        prediction = self.regression_layer(F.relu(glu))
        
        return prediction, reg_loss
