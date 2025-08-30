动手学LLaMA FA Quant GEMM ……
# Todo List
Todo顺序：Llama Transformer -> FlashAttention -> llama.cpp -> 量化 

把优化的gemm和pytorch实现对比一下性能？以及融合进框架里，看看能加速多少？

- [ ] llama系列
  - [ ] Transformer
    - [x] MHA
    - [x] RoPE
      - [x] 1. llama方式实现
      - [x] 2. 线性组合方式实现
      - [x] 1 vs 2 ：1比2快3.8倍左右（1023，2048，4096）平均测速。原因可能在于1通过复数乘法规避掉了2的transpose，stack，reshape这些中间过程，同时也避免了中间tensor的创建。
    - [ ] KVcache
    - [x] RMSNorm
    - [x] GQA
    - [x] LayerNorm
    - [x] BatchNorm
  - [ ] llama.cpp 【另外一个仓库】
  - [ ] llama3.1 【不了解】

- [ ] FlashAttention
  - [ ] python
  - [ ] cpp
  - [ ] cuda！！

- [ ] 量化
  - [ ] AutoAWQ
  - [ ] LLM.int8
  - [ ] GPTQ

- [ ] GEMM
  - [x] Cpp
  - [x] CUDA【0409】
  - [x] CUDA优化：warp优化（Shared Memory）对比
  - [ ] TensorRT 及其 Nsight System分析
    - [ ] 真的能自动安排grid/block这些吗？需要设置一个极端不合理的+TensorRT对比分析）
    - [ ] 对比量化前后，时间分析
  - [ ] Triton

# 文件说明

1. RoPE1.py：最简单的版本（基于llama）+维度变换注释

2. RoPE_llama.py：llama源码

3. RoPE_extra.py：研究外推性【训练max_seq_len=512，推理时候实际序列长度大于该值，模型精度？相关性？不变？不太懂】

