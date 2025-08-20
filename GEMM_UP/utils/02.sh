=== GEMM共享内存版本编译和运行脚本 ===
项目根目录: /root/LLM_d2l/GEMM_UP

=== 开始编译 ===
✅ 编译成功!

=== 开始运行 ===
执行文件: /root/LLM_d2l/GEMM_UP/bin/02_gemm_sharedmem

=== GPU Device Information ===
Device name: NVIDIA GeForce RTX 4090
Compute capability: 8.9
Total global memory: 24091 MB
Multiprocessor count: 128
Max threads per block: 1024

Matrix 512x512 - Grid: 4x4 blocks, Block: 16x16 threads
Matrix multiplication (512x512) * (512x512) completed in: 0.000263974 seconds
Performance: 1016.9 GFLOPS

Matrix 1024x1024 - Grid: 8x8 blocks, Block: 16x16 threads
Matrix multiplication (1024x1024) * (1024x1024) completed in: 0.000384034 seconds
Performance: 5591.91 GFLOPS

Matrix 2048x2048 - Grid: 16x16 blocks, Block: 16x16 threads
Matrix multiplication (2048x2048) * (2048x2048) completed in: 0.000746187 seconds
Performance: 23023.5 GFLOPS

Matrix 4096x4096 - Grid: 32x32 blocks, Block: 16x16 threads
Matrix multiplication (4096x4096) * (4096x4096) completed in: 0.00452767 seconds
Performance: 30355.4 GFLOPS