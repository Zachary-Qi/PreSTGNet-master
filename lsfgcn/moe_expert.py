import math
import torch
from torch import nn
import torch.nn.functional as F


class FFNExpert(nn.Module):
    def __init__(self, hidden_dim, dropout):   # LLM 进化之路， FFN 激活函数从 GELU -> SwiGLU
        super().__init__()  

        hidden_dim = hidden_dim
        mid_dim = hidden_dim * 4 // 3

        self.up = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.down = nn.Linear(mid_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, mid_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(
            self.down(
    
                F.silu(
                    self.gate(x)
                ) * self.up(x)
            )
        )
        return out


class BasicMOE(nn.Module):
    def __init__(self, feature_in, dropout, expert_number):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                FFNExpert(feature_in, dropout) for _ in range(expert_number)
            ]
        )
        self.gate = nn.Linear(feature_in, expert_number)
    
    def forward(self, x):
        expert_weight = F.softmax(self.gate(x), dim=1) 
        
        expert_out_list = [
            expert(x).unsqueeze(1) for expert in self.experts
        ]  

        expert_output = torch.cat(expert_out_list, dim=1)

        expert_weight = expert_weight.unsqueeze(1) # (batch, 1, expert_nuber)

        # expert_weight * expert_out_list
        output = torch.matmul(expert_weight, expert_output) # (batch, 1, feature_out)
        
        return output.squeeze()


class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k
    
    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)  # shape is (b * s, expert_number)
        
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  
        
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)
        
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )  
        expert_mask = expert_mask.permute(2, 1, 0)  
        
        return router_logits, router_weights, selected_experts, expert_mask


class SparseMOE(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k, dropout, shared_experts_number=2,):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.dropout = dropout
        self.shared_experts_number = shared_experts_number

        self.experts = nn.ModuleList(
            [
                FFNExpert(self.hidden_dim, self.dropout) for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()

        hidden_states = x.view(-1, hidden_dim) # shape is(b * s, hidden_dim)

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            router_weights_idx, top_x = torch.where(expert_mask[expert_idx]) 
            current_state = hidden_states.unsqueeze(
                0
            )[:, top_x, :].reshape(-1, hidden_dim) # （selected_token_number, hidden_dim）
            
            # current_state 
            # current_token_router_weight = router_weights[top_x, router_weights_idx].unsqueeze(-1) 
            
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[top_x, router_weights_idx].unsqueeze(-1)  # （selected_token_number, 1） 这里有广播

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits # shape 是 (b * s, expert_number)


class ShareExpertMOE(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k, dropout, shared_experts_number):
        super().__init__()

        self.routed_experts_moe = SparseMOE(hidden_dim, expert_number, top_k, dropout, shared_experts_number)
        self.shared_experts = nn.ModuleList(
            [
                FFNExpert(
                    hidden_dim, dropout
                ) for _ in range(shared_experts_number)
            ]
        )

    def forward(self, x):
        # x shape 是 (b, s, hidden_dim)
        sparse_moe_out, router_logits = self.routed_experts_moe(x)

        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ] 
        
        shared_experts_out = torch.stack(
            shared_experts_out, dim=0
        ).sum(dim=0, keepdim=False)
        
        return sparse_moe_out + shared_experts_out, router_logits

def switch_load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    
    router_probs = torch.softmax(router_logits, dim=-1)  # [b*s, num_experts]
    
    _, selected_experts = torch.topk(router_probs, k=2, dim=-1)  # [b*s]
    
    mask = torch.nn.functional.one_hot(selected_experts, num_experts).float()  # [b*s, num_experts]
    
    expected_load = torch.ones_like(router_probs) / num_experts
    
    actual_load = mask.mean(dim=0)  # [num_experts]
    
    aux_loss = torch.sum(actual_load * router_probs.mean(dim=0)) * num_experts
    
    z_loss = torch.mean(torch.square(router_logits))
    z_loss_weight = 0.001  
    
    total_loss = aux_loss + z_loss * z_loss_weight
    
    return total_loss

def test_moe_training():
    # Create a simple dataset
    batch_size = 32
    seq_len = 16
    hidden_dim = 32
    num_batches = 100
    
    # Initialize model and optimizer
    expert_number = 8 
    model = ShareExpertMOE(hidden_dim, expert_number, 2, 0.2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for batch in range(num_batches):
        # Generate random input data
        x = torch.randn(batch_size, seq_len, hidden_dim)
        target = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Forward pass
        output, router_logits = model(x)

        # Compute losses
        # MSE loss for prediction
        mse_loss = F.mse_loss(output, target)
        
        aux_loss = switch_load_balancing_loss(router_logits, expert_number)
        # Combined loss
        total_loss = mse_loss + 0.01 * aux_loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if batch % 10 == 0:
            print(f"Batch {batch}, Loss: {total_loss.item():.4f} "
                  f"(MSE: {mse_loss.item():.4f}, Aux: {aux_loss.item():.4f})")
