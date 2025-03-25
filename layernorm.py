import torch
import torch.nn as nn

# 创建输入张量，形状 [2, 4, 3]
input_tensor = torch.tensor([
    [[1, 3, 2],
     [0, 4, 3],
     [0, 1, 4],
     [2, 2, 2]],
    
    [[1, 1, 1],
     [2, 2, 4],
     [-1, 3, 1],
     [0, 5, 5]]
], dtype=torch.float32)

# 定义层归一化层，对最后一维（dim=2）做归一化
layer_norm = nn.LayerNorm(normalized_shape=3)

# 前向传播
output_tensor = layer_norm(input_tensor)

# 使用for循环实现层归一化
# LayerNorm：固定住batch和token，在dim上求均值和方差
def layer_norm_for_loop(input_tensor, eps=1e-6):
    # 获取输入张量的维度信息
    batch_size, seq_length, feature_size = input_tensor.shape

    # 初始化输出张量
    output_tensor = torch.zeros_like(input_tensor)
    # 对每个样本进行层归一化
    for i in range(batch_size):
        # 沿着最后一个维度，计算每个样本的均值和标准差
        # input_tensor[i] 第i个张量，沿着最后一个维度也就是列的方向求均值，同时保持ndim不变。(4,3)->(4,1)
        mean = torch.mean(input_tensor[i], dim=-1, keepdim=True)
        # 同理，沿着列的方向求方差
        std = torch.std(input_tensor[i], dim=-1, keepdim=True, unbiased=False)  # 一定要无偏估计
        # 进行归一化
        output_tensor[i] = (input_tensor[i] - mean) / (std + eps)
    return output_tensor

print("\n层归一化输出:")
print(output_tensor)

# 添加精度对比验证
diff = torch.abs(layer_norm(input_tensor) - output_tensor)
print(f"\n最大绝对误差: {diff.max().item():.2e}")
print(f"是否在1e-5误差范围内: {torch.allclose(layer_norm(input_tensor), output_tensor, atol=1e-5)}")

# 使用更严格的1e-6精度验证
print(f"是否在1e-6误差范围内: {torch.allclose(layer_norm(input_tensor), output_tensor, atol=1e-6)}")



