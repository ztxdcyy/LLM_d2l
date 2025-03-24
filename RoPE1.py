import numpy as np
import torch

def precompute_freqs_cis(max_seq_len, dim):
    # len(theta) = dim//2
    theta = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    index = np.arange(max_seq_len)
    # (max_seq_len,) (dim//2,) -> (max_seq_len, dim//2)
    freqs = torch.tensor(np.outer(index, theta))
    # 不变
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # （max_seq_len, head_dim//2） -> (1, max_seq_len, 1, head_dim//2)
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    # xq.shape = (batch, max_seq_len, num_heads, head_dim)
    # xq after reshape = (batch, max_seq_len, num_heads, head_dim//2, 2)
    # xq_.shape = (batch, max_seq_len, num_heads, head_dim//2)
    xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    # （max_seq_len, head_dim//2) -> (1, max_seq_len, 1, head_dim//2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # 逐元素相乘的时候shape不变
    # view_as_real:（batch, max_seq_len, num_heads, head_dim//2） -> (batch, max_seq_len, num_heads, head_dim//2, 2)
    # flatten(3)： (batch, max_seq_len, num_heads, head_dim//2, 2) -> (batch, max_seq_len, num_heads, head_dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 示例数据
batch = 3
max_seq_len = 512
dim = 768
num_heads = 8
head_dim = dim // num_heads
xq = torch.randn(batch, max_seq_len, num_heads, head_dim)
xk = torch.randn(batch, max_seq_len, num_heads, head_dim)

# 获取频率张量
freqs_cis = precompute_freqs_cis(max_seq_len, head_dim)

# 应用旋转位置编码
xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

print("xq after rotary embedding:", xq_out.shape)
print("xk after rotary embedding:", xk_out.shape)

"""
xq.shape = (batch, max_seq_len, num_heads, head_dim)
after reshape (batch, max_seq_len, num_heads, head_dim // 2, 2)
after view_as_complex (batch, max_seq_len, num_heads, head_dim//2) 抹掉最后一个2

freq_cis.shape = （max_seq_len, head_dim//2)
after reshape_for_broadcast shape= （1, max_seq_len, 1,  head_dim//2 )

xq_ * freqs_cis：* 代表 per element 相乘，需要经过broadcast
(batch, max_seq_len, num_heads, head_dim//2) * （1, max_seq_len, 1,  head_dim//2 ) -> (batch, max_seq_len, num_heads, head_dim//2) 

after view as real (batch, max_seq_len, num_heads, head_dim//2, 2) 添加上最后一个2
after flatten (batch, max_seq_len, num_heads, head_dim) 
"""