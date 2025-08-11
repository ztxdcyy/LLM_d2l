import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.w2 = nn.Linear(4 * d_model, d_model, bias=False)
        
    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 专家网络
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        
        # 门控网络
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x):
        # 1. 门控选择Top-K专家
        gate_logits = self.gate(x)  # [B, S, num_experts]
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [B, S, top_k]
        
        # 2. 初始化输出
        output = torch.zeros_like(x)  # [B, S, d_model]
        
        # 3. 逐个token处理
        for b in range(x.shape[0]):                             # 遍历batch
            for s in range(x.shape[1]):                         # 遍历sequence
                token_input = x[b, s, :]                        # [d_model]       当前token
                token_output = torch.zeros_like(token_input)    # [d_model]
                
                # 处理该token选中的top_k个专家
                for k in range(self.top_k):
                    expert_id = top_k_indices[b, s, k].item()   # 第k个选中的专家
                    weight = top_k_gates[b, s, k]               # 该专家的输出加权系数
                    
                    # 专家处理并加权累加
                    expert_out = self.experts[expert_id](token_input.unsqueeze(0))  # [1, d_model]
                    token_output += weight * expert_out.squeeze(0)                  # 加权累加
                
                output[b, s, :] = token_output
        
        return output

if __name__ == "__main__":
    # 测试
    batch_size, seq_len, d_model = 2, 4, 8
    num_experts, top_k = 4, 2
    
    moe = MoELayer(d_model, num_experts, top_k)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"输入: {x.shape}")
    output = moe(x)
    print(f"输出: {output.shape}")
    
    # 查看门控选择
    with torch.no_grad():
        gate_logits = moe.gate(x)
        _, indices = torch.topk(gate_logits, top_k, dim=-1)
        print(f"专家选择: {indices[0, 0]}")  # 第一个token选择的专家