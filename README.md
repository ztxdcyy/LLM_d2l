# Todo List
Todo顺序：Llama Transformer -> FlashAttention -> llama.cpp -> 量化 

把优化的gemm和pytorch实现对比一下性能？以及融合进框架里，跑端到端实验对比naive pytorch baseline

- [x] Llama
  - [x] Transformer
    - [x] MHA
    - [x] RoPE
      - [x] 1. llama方式实现
      - [x] 2. 线性组合方式实现
      - [x] 1 vs 2 ：1比2快3.8倍左右（1023，2048，4096）平均测速。原因可能在于1通过复数乘法规避掉了2的transpose，stack，reshape这些中间过程，同时也避免了中间tensor的创建。
    - [x] KVcache
    - [x] RMSNorm
    - [x] GQA
      - [ ] 支持 cudagraph 功能：https://github.com/fw-ai/llama-cuda-graph-example/commit/d8003f59af8893837ec9834c705cfd0035d3ad37
    - [x] LayerNorm
    - [x] BatchNorm


- [ ] FlashAttention
  - [ ] python
  - [ ] cpp
  - [ ] cuda！！

- [ ] GEMM
  - [x] Cpp
  - [x] CUDA【0409】
  - [x] CUDA优化：warp优化（Shared Memory）对比
  - [ ] TensorRT 及其 Nsight System分析
    - [ ] 真的能自动安排grid/block这些吗？需要设置一个极端不合理的+TensorRT对比分析）
    - [ ] 对比量化前后，时间分析
  - [ ] Triton

- [ ] Quant
  - [ ] AutoAWQ
  - [ ] LLM.int8
  - [ ] GPTQ



# 参考：

https://github.com/meta-pytorch/gpt-fast

https://github.com/zjhellofss/KuiperInfer

https://github.com/ifromeast/cuda_learning/blob/main/03_gemm/sgemm_v3.cu

