import numpy as np
import torch
import sys

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
    # 确保数据在GPU上
    xq = xq.cuda()
    xk = xk.cuda()
    freqs_cis = freqs_cis.cuda()
    
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
    start_time = time.time()
    # 使用CUDA加速计算
    with torch.amp.autocast(device_type='cuda'):
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    torch.cuda.synchronize()  # 确保CUDA操作完成
    elapsed = time.time() - start_time
    
    return xq_out.type_as(xq), xk_out.type_as(xk), elapsed


if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) != 5:
        print("用法: python RoPE1.py batch_size seq_len num_heads head_dim")
        sys.exit(1)
        
    batch = int(sys.argv[1])
    max_seq_len = int(sys.argv[2])
    num_heads = int(sys.argv[3])
    head_dim = int(sys.argv[4])
    
    xq = torch.randn(batch, max_seq_len, num_heads, head_dim)
    xk = torch.randn(batch, max_seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(max_seq_len, head_dim)
    
    xq_out, xk_out, elapsed = apply_rotary_emb(xq, xk, freqs_cis)
    print(elapsed)  


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


"""
设复数形式：q_complex = q_real + i*q_imag
旋转操作：q_rotated = q_complex * (cosθ + i*sinθ)
展开实数形式：
q_real' = q_real*cosθ - q_imag*sinθ
q_imag' = q_real*sinθ + q_imag*cosθ
"""