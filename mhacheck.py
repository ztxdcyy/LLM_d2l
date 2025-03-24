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

# 初始化模型
mha = MHA.MHA2(d_model, num_heads)
# https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
mha_pytorch = nn.MultiheadAttention(
    embed_dim=d_model, 
    num_heads=num_heads, 
    batch_first=True,
    bias=False)

# 打印自己实现的 MHA 中 W_qkv 和 W_o 的参数
print("自定义 MHA 的 W_qkv 权重:", mha.W_qkv.weight.data)
print("自定义 MHA 的 W_qkv 偏置:", mha.W_qkv.bias.data)
print("自定义 MHA 的 W_o 权重:", mha.W_o.weight.data)
print("自定义 MHA 的 W_o 偏置:", mha.W_o.bias.data)

# 打印 PyTorch 内置 MHA 的参数
print("PyTorch MHA 的 in_proj_weight:", mha_pytorch.in_proj_weight.data)
# print("PyTorch MHA 的 in_proj_bias:", mha_pytorch.in_proj_bias.data)
print("PyTorch MHA 的 out_proj 权重:", mha_pytorch.out_proj.weight.data)
# print("PyTorch MHA 的 out_proj 偏置:", mha_pytorch.out_proj.bias.data)

# 计算输出，我的mha直接输入x，qkv的embedding在mha内部实现，而pytorch的输入是q,k,v，需要把embedding拿出来。
output, attn = mha(Q, K, V)
print("自定义 MHA 的 Q 形状:", Q.shape)
print("自定义 MHA 的 K 形状:", K.shape)
print("自定义 MHA 的 V 形状:", V.shape)
print("自定义 MHA 的 attn 形状:", attn.shape)
print("自定义 MHA 的 attn 部分值:", attn[0, 0, :5, :5])
print("自定义 MHA 的 output 形状:", output.shape)
print("自定义 MHA 的 output 部分值:", output[0, :5, :5])

output_pytorch, attn_pytorch = mha_pytorch(Q, K, V)
print("PyTorch MHA 的 attn_pytorch 形状:", attn_pytorch.shape)
print("PyTorch MHA 的 attn_pytorch 部分值:", attn_pytorch[0, :5, :5])
print("PyTorch MHA 的 output_pytorch 形状:", output_pytorch.shape)
print("PyTorch MHA 的 output_pytorch 部分值:", output_pytorch[0, :5, :5])

# 验证输出是否一致
print(torch.allclose(output, output_pytorch, atol=1e-6))
