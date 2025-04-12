import numpy as np
import torch

def precompute_freqs_cis(max_seq_len, dim):
    theta = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    index = np.arange(max_seq_len)
    freqs = torch.tensor(np.outer(index, theta))
    # freq_cis.shape = (max_seq_len, head_dim//2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def RoPE(xq, xk):
    # 确保数据在GPU上
    xq = xq.cuda()
    xk = xk.cuda()
    batch, seq_len, num_heads, head_dim = xq.shape

    start_time = time.time()
    
    # 预计算频率
    freqs_cis = precompute_freqs_cis(seq_len, head_dim).to(xq.device)
    freqs_real = torch.view_as_real(freqs_cis)
    
    # 分离cos和sin, 最后一个维度每个重复一次，[seq_len, head_dim//2] -> [seq_len, head_dim]
    cos = freqs_real[..., 1].unsqueeze(-1).expand(-1, -1, 2).reshape(seq_len, head_dim)
    sin = freqs_real[..., 0].unsqueeze(-1).expand(-1, -1, 2).reshape(seq_len, head_dim)
    
    # 调整维度并广播
    xq = xq.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    xk = xk.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]

    with torch.cuda.amp.autocast():
        # 计算q的旋转
        xq_rot = torch.stack([-xq[..., 1::2], xq[..., ::2]], dim=-1).reshape(batch, num_heads, seq_len, head_dim)
        # start_time = time.time()
        xq_out = xq * cos + xq_rot * sin
        # elapsed = time.time() - start_time
        
        # 计算k的旋转
        xk_rot = torch.stack([-xk[..., 1::2], xk[..., ::2]], dim=-1).reshape(batch, num_heads, seq_len, head_dim)
        xk_out = xk * cos + xk_rot * sin
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    return xq_out.transpose(1, 2), xk_out.transpose(1, 2), elapsed

if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) != 5:
        print("用法: python RoPE2.py batch_size seq_len num_heads head_dim")
        sys.exit(1)
        
    batch = int(sys.argv[1])
    max_seq_len = int(sys.argv[2])
    num_heads = int(sys.argv[3])
    head_dim = int(sys.argv[4])
    
    xq = torch.randn(batch, max_seq_len, num_heads, head_dim)
    xk = torch.randn(batch, max_seq_len, num_heads, head_dim)
    
    # 直接计算q和k并计时
    xq_out, xk_out, total_time = RoPE(xq, xk)
    print(total_time)