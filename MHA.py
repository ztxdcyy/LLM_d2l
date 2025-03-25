import torch
import torch.nn as nn

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

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True) + self.eps
        x = x * torch.rsqrt(norm)
        return self.weight * x
    
class MHA_RMSNorm (nn.Module):
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
  

if __name__ == "__main__":
  mha = MHA(768, 8)
  x = torch.randn(3, 512, 768)
  y, attn = mha(x)
  print(y.shape)