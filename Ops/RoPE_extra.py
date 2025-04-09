import numpy as np
import torch

# 计算频率基础值
def get_freqs(max_seq_len, dim):
    theta = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.outer(np.arange(max_seq_len), theta)
    return freqs

# 计算频率张量
def get_freqs_cis(max_seq_len, dim):
    freqs = get_freqs(max_seq_len, dim)
    freqs_cis = np.cos(freqs) + 1j * np.sin(freqs)
    return freqs_cis

# 应用旋转位置编码
def apply_rotary_emb(x, freqs_cis):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    cos, sin = freqs_cis.real, freqs_cis.imag
    cos = torch.tensor(np.repeat(cos, 2, axis=-1), dtype=x.dtype)
    sin = torch.tensor(np.repeat(sin, 2, axis=-1), dtype=x.dtype)
    return x * cos + rotate_half(x) * sin

# 训练时的序列长度
train_max_seq_len = 128
dim = 64
# 推理时的序列长度，比训练时的长
test_max_seq_len = 256

# 生成训练时的频率张量
train_freqs_cis = get_freqs_cis(train_max_seq_len, dim)

# 模拟训练时的输入
x_train = torch.randn(1, train_max_seq_len, dim)
x_train_rope = apply_rotary_emb(x_train, train_freqs_cis)

# 生成推理时的频率张量
test_freqs_cis = get_freqs_cis(test_max_seq_len, dim)

# 模拟推理时的输入，序列长度更长
x_test = torch.randn(1, test_max_seq_len, dim)
x_test_rope = apply_rotary_emb(x_test, test_freqs_cis)

print(f"训练时序列长度: {train_max_seq_len}, 应用 RoPE 后的形状: {x_train_rope.shape}")
print(f"推理时序列长度: {test_max_seq_len}, 应用 RoPE 后的形状: {x_test_rope.shape}")
