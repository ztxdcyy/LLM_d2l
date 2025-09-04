各个版本在4090上的对比（离开了H20……）

我们统一观测：MNK=4096，BM=BK=128, TM=TN=8, dim3 block(BM/TM, BK/TK)

sgemm_v123来自于紫气东来

（https://www.zhihu.com/people/zi-qi-dong-lai-1）

（https://github.com/ifromeast/cuda_learning/blob/main/03_gemm/）

c，开始慌了，咋都突破硬件上限了，太抽象了，等会验证下正确性。

4090查得到的数据FP32 (float) 82.58 TFLOPS 我那个不知道咋回事都120多TFLOPs了，抽象。


# 数据记录
## 01 5089.96 GFLOPS（100%）
```
root@autodl-container-3f7446bd49-385cc1ca:~/LLM_d2l/GEMM_UP# ./utils/01.sh 
Compiling src/01_gemm_blocksize.cu...
Compilation successful!

==========================================
Testing with blocksize: 8x8
==========================================
=== GPU Device Information ===
Device name: NVIDIA GeForce RTX 4090
Compute capability: 8.9
Total global memory: 24091 MB
Multiprocessor count: 128
Max threads per block: 1024
Max grid size: 2147483647 x 65535 x 65535
Max block dimensions: 1024 x 1024 x 64
===============================

Using block size: 8x8 (64 threads per block)
Maximum supported square blocksize: 32

Matrix 512x512 - Grid: 64x64 blocks, Block: 8x8 threads
Matrix multiplication (512x512) completed in: 0.0175218 seconds
Performance: 15.3201 GFLOPS

Matrix 1024x1024 - Grid: 128x128 blocks, Block: 8x8 threads
Matrix multiplication (1024x1024) completed in: 0.000589082 seconds
Performance: 3645.47 GFLOPS

Matrix 2048x2048 - Grid: 256x256 blocks, Block: 8x8 threads
Matrix multiplication (2048x2048) completed in: 0.0042951 seconds
Performance: 3999.88 GFLOPS

Matrix 4096x4096 - Grid: 512x512 blocks, Block: 8x8 threads
Matrix multiplication (4096x4096) completed in: 0.038389 seconds
Performance: 3580.17 GFLOPS


==========================================
Testing with blocksize: 16x16
==========================================

Using block size: 16x16 (256 threads per block)
Maximum supported square blocksize: 32

Matrix 512x512 - Grid: 32x32 blocks, Block: 16x16 threads
Matrix multiplication (512x512) completed in: 0.000262491 seconds
Performance: 1022.65 GFLOPS

Matrix 1024x1024 - Grid: 64x64 blocks, Block: 16x16 threads
Matrix multiplication (1024x1024) completed in: 0.000461509 seconds
Performance: 4653.18 GFLOPS

Matrix 2048x2048 - Grid: 128x128 blocks, Block: 16x16 threads
Matrix multiplication (2048x2048) completed in: 0.0034285 seconds
Performance: 5010.9 GFLOPS

Matrix 4096x4096 - Grid: 256x256 blocks, Block: 16x16 threads
Matrix multiplication (4096x4096) completed in: 0.027002 seconds
Performance: 5089.96 GFLOPS


==========================================
Testing with blocksize: 32x32
==========================================

Using block size: 32x32 (1024 threads per block)
Maximum supported square blocksize: 32

Matrix 512x512 - Grid: 16x16 blocks, Block: 32x32 threads
Matrix multiplication (512x512) completed in: 0.000206874 seconds
Performance: 1297.58 GFLOPS

Matrix 1024x1024 - Grid: 32x32 blocks, Block: 32x32 threads
Matrix multiplication (1024x1024) completed in: 0.000464865 seconds
Performance: 4619.59 GFLOPS

Matrix 2048x2048 - Grid: 64x64 blocks, Block: 32x32 threads
Matrix multiplication (2048x2048) completed in: 0.00340109 seconds
Performance: 5051.28 GFLOPS

Matrix 4096x4096 - Grid: 128x128 blocks, Block: 32x32 threads
Matrix multiplication (4096x4096) completed in: 0.0269797 seconds
Performance: 5094.16 GFLOPS


==========================================
Testing with blocksize: 64x64
==========================================
Error: blocksize 64x64 (4096 threads) exceeds device limit of 1024 threads per block
Maximum square blocksize for this device: 32

All tests completed!
```

## 02 30355.4 GFLOPS（5.9637）

4090 需要在utils02sh里把sm改成89 

```
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
```

## 03 121502 GFLOPS （23.87091）

这个版本有问题，性能突破了物理上限，计算结果也是错的离谱。

```
root@autodl-container-3f7446bd49-385cc1ca:~/LLM_d2l/GEMM_UP# ./utils/03.sh 
=== GEMM共享内存版本编译和运行脚本 ===
项目根目录: /root/LLM_d2l/GEMM_UP

=== 开始编译 ===
=== 开始运行 ===
执行文件: /root/LLM_d2l/GEMM_UP/bin/03

=== GPU Device Information ===
Device name: NVIDIA GeForce RTX 4090
Compute capability: 8.9
Total global memory: 24091 MB
Multiprocessor count: 128
Max threads per block: 1024

Matrix 512x512 - Grid: 4x4 blocks, Block: 16x16 threads
Matrix multiplication (512x512) * (512x512) completed in: 0.000116361 seconds
Performance: 2306.92 GFLOPS

Matrix 1024x1024 - Grid: 8x8 blocks, Block: 16x16 threads
Matrix multiplication (1024x1024) * (1024x1024) completed in: 8.0301e-05 seconds
Performance: 26742.9 GFLOPS

Matrix 2048x2048 - Grid: 16x16 blocks, Block: 16x16 threads
Matrix multiplication (2048x2048) * (2048x2048) completed in: 0.000184432 seconds
Performance: 93150.2 GFLOPS

Matrix 4096x4096 - Grid: 32x32 blocks, Block: 16x16 threads
Matrix multiplication (4096x4096) * (4096x4096) completed in: 0.00113117 seconds
Performance: 121502 GFLOPS

=== 脚本执行完成 ===
```

```
(base) root@autodl-container-10be4ca793-400c39a3:~/workspace/LLM_d2l/GEMM_UP# ./utils/03.sh 
=== GEMM共享内存版本编译和运行脚本 ===
项目根目录: /root/workspace/LLM_d2l/GEMM_UP

=== 开始编译 ===
=== 开始运行 ===
执行文件: /root/workspace/LLM_d2l/GEMM_UP/bin/03

=== GPU Device Information ===
Device name: NVIDIA GeForce RTX 4090
Compute capability: 8.9
Total global memory: 24091 MB
Multiprocessor count: 128
Max threads per block: 1024

Matrix 512x512 - Grid: 4x4 blocks, Block: 16x16 threads
Sampled max abs error: 131.09
Matrix multiplication (512x512) * (512x512) completed in: 3.2325e-05 seconds
Performance: 8304.27 GFLOPS

Matrix 1024x1024 - Grid: 8x8 blocks, Block: 16x16 threads
Sampled max abs error: 265.641
Matrix multiplication (1024x1024) * (1024x1024) completed in: 5.6046e-05 seconds
Performance: 38316.4 GFLOPS

Matrix 2048x2048 - Grid: 16x16 blocks, Block: 16x16 threads
Sampled max abs error: 533.071
Matrix multiplication (2048x2048) * (2048x2048) completed in: 0.000163538 seconds
Performance: 105051 GFLOPS

Matrix 4096x4096 - Grid: 32x32 blocks, Block: 16x16 threads
Sampled max abs error: 1044.09
Matrix multiplication (4096x4096) * (4096x4096) completed in: 0.00111716 seconds
Performance: 123025 GFLOPS


=== 脚本执行完成 ===
```

## 03 v2

```
(base) root@autodl-container-10be4ca793-400c39a3:~/workspace/LLM_d2l/GEMM_UP# ./utils/03v2.sh 
=== GEMM共享内存版本编译和运行脚本 ===
项目根目录: /root/workspace/LLM_d2l/GEMM_UP

=== 开始编译 ===
=== 开始运行 ===
执行文件: /root/workspace/LLM_d2l/GEMM_UP/bin/03

=== GPU Device Information ===
Device name: NVIDIA GeForce RTX 4090
Compute capability: 8.9
Total global memory: 24091 MB
Multiprocessor count: 128
Max threads per block: 1024

Matrix 512x512 - Grid: 4x4, Block: 16x16
Sampled max abs error: 6.48024e-05
Avg time: 6.48512e-05 s, Performance: 3854.98 GiFLOPS

Matrix 1024x1024 - Grid: 8x8, Block: 16x16
Sampled max abs error: 0.0001934
Avg time: 0.000126019 s, Performance: 15870.6 GiFLOPS

Matrix 2048x2048 - Grid: 16x16, Block: 16x16
Sampled max abs error: 0.000835282
Avg time: 0.000429402 s, Performance: 37261.2 GiFLOPS

Matrix 4096x4096 - Grid: 32x32, Block: 16x16
Sampled max abs error: 0.00238856
Avg time: 0.00332798 s, Performance: 38461.8 GiFLOPS


=== 脚本执行完成 ===
```

## sgemm v1
```
root@autodl-container-3f7446bd49-385cc1ca:~/LLM_d2l/GEMM_UP# ./utils/sgemm_v1.sh 
=== GEMM共享内存版本编译和运行脚本 ===
项目根目录: /root/LLM_d2l/GEMM_UP

=== 开始编译 ===
=== 开始运行 ===
执行文件: /root/LLM_d2l/GEMM_UP/bin/sgemm_v1


Kernal = sgemm_V1
Max Error = 0.000046
M N K =    128    128   1024, Time =   0.00012902   0.00013031   0.00013718 s, AVG Performance =   239.8120 Gflops
M N K =    192    192   1024, Time =   0.00012800   0.00012922   0.00013094 s, AVG Performance =   544.1335 Gflops
M N K =    256    256   1024, Time =   0.00012902   0.00012914   0.00013005 s, AVG Performance =   967.9238 Gflops
M N K =    384    384   1024, Time =   0.00012880   0.00012932   0.00013008 s, AVG Performance =  2174.9183 Gflops
M N K =    512    512   1024, Time =   0.00012800   0.00012904   0.00013005 s, AVG Performance =  3874.8636 Gflops
M N K =    768    768   1024, Time =   0.00012902   0.00013006   0.00013414 s, AVG Performance =  8650.0135 Gflops
M N K =   1024   1024   1024, Time =   0.00013107   0.00013204   0.00013517 s, AVG Performance = 15147.4761 Gflops
M N K =   1536   1536   1024, Time =   0.00023347   0.00023448   0.00023645 s, AVG Performance = 19191.1402 Gflops
M N K =   2048   2048   1024, Time =   0.00024154   0.00024294   0.00024576 s, AVG Performance = 32929.8332 Gflops
M N K =   3072   3072   1024, Time =   0.00057344   0.00057619   0.00057856 s, AVG Performance = 31239.7616 Gflops
M N K =   4096   4096   1024, Time =   0.00091878   0.00092424   0.00092877 s, AVG Performance = 34623.0421 Gflops
M N K =   6144   6144   1024, Time =   0.00200602   0.00201142   0.00201421 s, AVG Performance = 35795.5918 Gflops
M N K =   8192   8192   1024, Time =   0.00353674   0.00354127   0.00354525 s, AVG Performance = 36145.1877 Gflops
M N K =  12288  12288   1024, Time =   0.00787664   0.00789032   0.00790938 s, AVG Performance = 36500.4354 Gflops
M N K =  16384  16384   1024, Time =   0.01424998   0.01427649   0.01429504 s, AVG Performance = 35863.1578 Gflops

=== 脚本执行完成 ===
```
