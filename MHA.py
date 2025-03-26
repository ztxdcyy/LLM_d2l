import torch
import torch.nn as nn
from typing import Optional, Tuple
import RoPE1
import math

class ModelArgs:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None        # 默认为0，代表默认才用MHA，当n_kv_heads<n_heads时，代表使用GQA
    norm_eps: float = 1e-5

    max_batch_size: int = 32      # 这两个max均是为了kvcache预留的空间
    max_seq_len: int = 2048

# 输入是x的版本
class MHA (nn.Module):
  def __init__(self, d_model, num_heads):
    super(MHA, self).__init__()
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    self.W_qkv = nn.Linear(d_model, 3*d_model)
    self.W_o = nn.Linear(d_model, d_model)

  def attention(self, query, key, value, mask = None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
    if mask is not None:
      scores = scores.masked_fill(mask==0, 1e-9)
    p_attn = torch.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

  def split_heads(self, x):
    batch, seqlength, d_model = x.size()
    # 想想为什么要交换维度呢？主要是将（batch,num_heads）放在前面，后面的（seqlength, d_k）是每个头上的qkv的shape
    return x.view(batch, seqlength, self.num_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
    batch, num_heads, seqlength, d_k = x.size()
    return x.transpose(1, 2).contiguous().view(batch, seqlength, self.d_model)

  def forward(self, x, mask=None):
    # nn.Linear Linear接受的参数是(in_feat， out_feat) x.shape = (b, l, d_model)
    # qkv.shape = (b, l, 3d_model)
    qkv = self.W_qkv(x)
    # split沿着最后一个维度拆分，每个都拆分成d_model大小
    Q, K, V = torch.split(qkv, self.d_model, dim=-1)

    # Q.shape = （batch，numheads，seqlength，dk）
    Q = self.split_heads(Q)
    K = self.split_heads(K)
    V = self.split_heads(V)

    x, attn = self.attention(Q, K, V)
    x = self.combine_heads(x)

    return self.W_o(x), attn
  
# 输入是qkv，而不是x
class MHA2 (nn.Module):
  def __init__(self, d_model, num_heads):
    super(MHA2, self).__init__()
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    self.W_qkv = nn.Linear(d_model, 3*d_model)
    self.W_o = nn.Linear(d_model, d_model)

  def attention(self, query, key, value, mask = None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
    if mask is not None:
      scores = scores.masked_fill(mask==0, 1e-9)
    p_attn = torch.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

  def split_heads(self, x):
    batch, seqlength, d_model = x.size()
    # 想想为什么要交换维度呢？主要是将（batch,num_heads）放在前面，后面的（seqlength, d_k）是每个头上的qkv的shape
    return x.view(batch, seqlength, self.num_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
    batch, num_heads, seqlength, d_k = x.size()
    return x.transpose(1, 2).contiguous().view(batch, seqlength, self.d_model)

  def forward(self, Q, K, V, mask=None):  
    # （batch，seqlength，d_model）->（batch，head_num，seqlength，head_dim）
    Q = self.split_heads(Q)
    K = self.split_heads(K)
    V = self.split_heads(V)

    x, attn = self.attention(Q, K, V)
    x = self.combine_heads(x)

    return self.W_o(x), attn

# KVcache + RoPE + GQA
class GQA (nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    self.n_rep = args.n_heads // self.n_kv_heads
    self.head_dim = args.dim // args.n_heads
    self.head_num = args.n_heads

    self.wq = nn.Linear(args.dim, self.head_num * self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    # 创建一个固定shape的张量用于存储kvcache
    self.cache_k = torch.zeros(
        (
            args.max_batch_size,
            args.max_seq_len,
            self.n_kv_heads,
            self.head_dim,
        )
    ).cuda()
    self.cache_v = torch.zeros(
        (
            args.max_batch_size,
            args.max_seq_len,
            self.n_kv_heads,
            self.head_dim,
        )
    ).cuda()

  def forward(
      self,
      x: torch.Tensor,
      start_pos: int,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor]
    ):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.head_num, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = RoPE1.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    # 统一下设备，万一xq是cpu上的，就需要统一下
    self.cache_k = self.cache_k.to(xq)
    self.cache_v = self.cache_v.to(xq)

    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
    self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

    keys = self.cache_k[:bsz, : start_pos + seqlen]
    values = self.cache_v[:bsz, : start_pos + seqlen]

    keys = keys.view(bsz, -1, self.n_kv_heads, self.head_dim)
    values = values.view(bsz, -1, self.n_kv_heads, self.head_dim)

    xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
    values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None:
        scores = scores + mask  # (bs, n_heads, seqlen, cache_len + seqlen)
    scores = torch.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)
  

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True) + self.eps
        # rsqrt，计算张量每个元素开根号的倒数
        x = x * torch.rsqrt(norm)
        return self.weight * x

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff):
    super(FeedForward, self).__init__()
    self.W_1 = nn.Linear(d_model, d_ff)
    self.W_2 = nn.Linear(d_ff, d_model)
  def forward(self, x):
    return self.W_2(torch.relu(self.W_1(x)))

class MHABlock(nn.Module):
  def __init__(self, d_model, num_heads):
    super(MHABlock, self).__init__()
    self.mha = MHA(d_model, num_heads)
    self.ffn = FeedForward(d_model, 4*d_model)
    self.norm1 = RMSNorm(d_model)
    self.norm2 = RMSNorm(d_model)

  def forward(self, x):
    x = x + self.mha(self.norm1(x))[0]    # mha会返回两个值，一个是输出，一个是注意力权重
    x = x + self.ffn(self.norm2(x))       # pre Norm
    return x  

class myTransformer(nn.module):
  def __init__(self, args: ModelArgs):
    super(myTransformer, self).__init__()
    self.embedding = nn.Embedding(args.vocab_size, args.d_model)
    self.pos_embedding = nn.Embedding(args.max_seq_length, args.d_model)
    

if __name__ == "__main__":
  mha = MHA(768, 8)
  x = torch.randn(3, 512, 768)
  y, attn = mha(x)
  print(y.shape)