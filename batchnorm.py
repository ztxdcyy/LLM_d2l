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

input_tensor = input_tensor.transpose(1, 2).view(2, 3, 2, 2)
batch_norm = nn.BatchNorm2d(num_features=3)

# 前向传播
output_tensor = batch_norm(input_tensor)

# BatchNorm：固定住dim，在batch和token（NHW）上求均值和方差
def batch_norm_for_loop(input_tensor, eps=1e-6):
    # 获取输入张量的维度信息 (N, C, H, W)
    batch_size, num_channels, height, width = input_tensor.shape
    output_tensor = torch.zeros_like(input_tensor)
    
    for c in range(num_channels):
        # 提取当前通道（channel=c）的特征图 [N, H, W]
        channel_data = input_tensor[:, c, :, :]
        
        # 沿批量（N）和空间维度（H, W）计算均值/方差
        # 该data内的所有元素都计算均值方差，得到的是标量
        mean = channel_data.mean(dim=(0, 1, 2), keepdim=True)  # 正确维度：N+H+W
        # print(mean.shape)
        std = channel_data.std(dim=(0, 1, 2), keepdim=True, unbiased=False)
        # print(std.shape)
        
        # 归一化并写回结果
        output_tensor[:, c, :, :] = (channel_data - mean) / (std + eps)
    
    return output_tensor

print("\n批量归一化输出:")
print(batch_norm_for_loop(input_tensor))
# 添加精度对比验证
diff = torch.abs(batch_norm_for_loop(input_tensor) - output_tensor)
print(f"\n最大绝对误差: {diff.max().item():.2e}")
print(f"是否在1e-5误差范围内: {torch.allclose(batch_norm_for_loop(input_tensor), output_tensor, atol=1e-5)}")
# 使用更严格的1e-6精度验证
print(f"是否在1e-6误差范围内: {torch.allclose(batch_norm_for_loop(input_tensor), output_tensor, atol=1e-6)}")
