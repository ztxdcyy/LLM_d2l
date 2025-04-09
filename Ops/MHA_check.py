# 验证pytorch调用的mha和自己写的结果是否一致，使用torch.allclose

import torch
import torch.nn as nn
import MHA

# 初始参数及随机数种子固定
torch.manual_seed(42)
batch = 3
seqlength = 512
d_model = 768
num_heads = 8

# 初始化输入数据
x = torch.randn(batch, seqlength, d_model)
W_qkv = nn.Linear(d_model, 3*d_model)
qkv = W_qkv(x)
Q, K, V = torch.split(qkv, d_model, dim=-1)
W_o = nn.Linear(d_model, d_model)

# 初始化模型
mha = MHA.MHA2(d_model, num_heads)
# https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
mha_pytorch = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    batch_first=True,
    bias=False,
    kdim=d_model,  # 添加kdim和vdim定义
    vdim=d_model
)

with torch.no_grad():
    # 替换为Parameter类型
    mha_pytorch.in_proj_weight = nn.Parameter(torch.empty(0))  # 使用Parameter包装
    mha_pytorch.q_proj_weight = None
    mha_pytorch.k_proj_weight = None
    mha_pytorch.v_proj_weight = None
    # 输出层参数设置保持不变
    # mha_pytorch.out_proj.weight.copy_(W_o.weight)
    # mha_pytorch.out_proj.bias.copy_(W_o.bias)


# 计算输出，我的mha直接输入x，qkv的embedding在mha内部实现，而pytorch的输入是q,k,v，需要把embedding拿出来。
output, attn = mha(Q, K, V)
output_pytorch, attn_pytorch = mha_pytorch(Q, K, V)

# 添加参数验证
print("Q矩阵一致:", torch.allclose(mha.W_q.weight, mha_pytorch.q_proj_weight))
print("输出差异:", torch.max(torch.abs(output - output_pytorch)))

# 验证输出是否一致
print(torch.allclose(output, output_pytorch, atol=1e-6))