import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    # 生成一个维度为dim//2的tensor，假如dim不是2的整数倍，最后还会切片成整数倍。
    # 1/10000**[2（j-1）/dim], j = [1, dim//2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个索引序列【0， end】 end就是max-seq-len
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # 外积，outer((m,),(n,)) = (m,n) 前面一个向量做转置之后，（m,1）*(1,n) = (m,n)
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # torch.polar(abs, angle) 返回极坐标形式的复数
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    # ndim： Number of dim，维度的个数，就是几维的张量
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # freqs_cis.shape = (max_seq_len, head_dim//2)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    # xq.shape = (batch_size, seq_len, num_heads, head_dim) 
    # reshape前后，元素总数量保持一致，-1的位置代表自动计算，2代表强制最后一个维度为2。shape=(batch_size, seq_len, num_heads, head_dim//2, 2)
    # view_as_complex将最后一个维度拆分成两个维度，分别代表实部和虚部，会把最后一个维度2unsqueeze掉，xq_.shape=(batch_size, seq_len, num_heads, head_dim//2)
    # view_as_complex就是最后一个维度2，缩掉。viewasreal就是扩出来最后一个维度2，强制。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 将freqs_cis的维度调整为xq_的维度，freqs_cis[0]作为新向量的[1]，freqs_cis[1]作为新向量的[-1]，其他的就是1
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # RoPE 的核心公式Re【xq*e^{i*theta*m}】，这里e指数已经被欧拉公式展开了，theta也计算完了
    # 这里其实涉及到pytorch的broadcast的概念
    # 首先俩张量得兼容，兼容就是该维度为1，或者俩维度相等。
    # 然后广播的意思就是，如果俩维度不相等，那么就会把为1的那个维度复制成大的那个维度
    # 然后这里的“*”代表的计算是逐个元素对于位置相乘，所以输出的shape维度和xq_保持一致，shape=(batch_size, seq_len, num_heads, head_dim//2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    # 经过view_as_real (b, l, num_heads, dk//2) -> (b, l, num_heads, dk//2, 2)。本来是复数，实部虚部拆开，最后多了一个维度，2。
    # 然后flatten(3)，把最后一个维度的2合并到倒数第二个维度，shape=(batch_size, seq_len, num_heads, head_dim) head_dim也就是dk，就是每个头上的向量的长度。
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)