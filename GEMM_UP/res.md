```
root@k8s-node07:/sgl-workspace/LLM_d2l/GEMM_UP# ./utils/test_blocksize.sh 
Compiling src/01_gemm_blocksize.cu...
Compilation successful!

==========================================
Testing with blocksize: 8x8
==========================================
=== GPU Device Information ===
Device name: NVIDIA H20-3e
Compute capability: 9.0
Total global memory: 143167 MB
Multiprocessor count: 78
Max threads per block: 1024
Max grid size: 2147483647 x 65535 x 65535
Max block dimensions: 1024 x 1024 x 64
===============================

Using block size: 8x8 (64 threads per block)
Maximum supported square blocksize: 32

Matrix 512x512 - Grid: 64x64 blocks, Block: 8x8 threads
Matrix multiplication (512x512) completed in: 0.018987 seconds
Performance: 14.1378 GFLOPS

Matrix 1024x1024 - Grid: 128x128 blocks, Block: 8x8 threads
Matrix multiplication (1024x1024) completed in: 0.00113796 seconds
Performance: 1887.14 GFLOPS

Matrix 2048x2048 - Grid: 256x256 blocks, Block: 8x8 threads
Matrix multiplication (2048x2048) completed in: 0.00877435 seconds
Performance: 1957.96 GFLOPS

Matrix 4096x4096 - Grid: 512x512 blocks, Block: 8x8 threads
Matrix multiplication (4096x4096) completed in: 0.069766 seconds
Performance: 1970 GFLOPS


==========================================
Testing with blocksize: 16x16
==========================================
=== GPU Device Information ===
Device name: NVIDIA H20-3e
Compute capability: 9.0
Total global memory: 143167 MB
Multiprocessor count: 78
Max threads per block: 1024
Max grid size: 2147483647 x 65535 x 65535
Max block dimensions: 1024 x 1024 x 64
===============================

Using block size: 16x16 (256 threads per block)
Maximum supported square blocksize: 32

Matrix 512x512 - Grid: 32x32 blocks, Block: 16x16 threads
Matrix multiplication (512x512) completed in: 0.0188691 seconds
Performance: 14.2262 GFLOPS

Matrix 1024x1024 - Grid: 64x64 blocks, Block: 16x16 threads
Matrix multiplication (1024x1024) completed in: 0.000703556 seconds
Performance: 3052.33 GFLOPS

Matrix 2048x2048 - Grid: 128x128 blocks, Block: 16x16 threads
Matrix multiplication (2048x2048) completed in: 0.00538987 seconds
Performance: 3187.43 GFLOPS

Matrix 4096x4096 - Grid: 256x256 blocks, Block: 16x16 threads
Matrix multiplication (4096x4096) completed in: 0.0440873 seconds
Performance: 3117.43 GFLOPS


==========================================
Testing with blocksize: 32x32
==========================================
=== GPU Device Information ===
Device name: NVIDIA H20-3e
Compute capability: 9.0
Total global memory: 143167 MB
Multiprocessor count: 78
Max threads per block: 1024
Max grid size: 2147483647 x 65535 x 65535
Max block dimensions: 1024 x 1024 x 64
===============================

Using block size: 32x32 (1024 threads per block)
Maximum supported square blocksize: 32

Matrix 512x512 - Grid: 16x16 blocks, Block: 32x32 threads
Matrix multiplication (512x512) completed in: 0.018898 seconds
Performance: 14.2044 GFLOPS

Matrix 1024x1024 - Grid: 32x32 blocks, Block: 32x32 threads
Matrix multiplication (1024x1024) completed in: 0.000616194 seconds
Performance: 3485.08 GFLOPS

Matrix 2048x2048 - Grid: 64x64 blocks, Block: 32x32 threads
Matrix multiplication (2048x2048) completed in: 0.00455601 seconds
Performance: 3770.82 GFLOPS

Matrix 4096x4096 - Grid: 128x128 blocks, Block: 32x32 threads
Matrix multiplication (4096x4096) completed in: 0.0381275 seconds
Performance: 3604.72 GFLOPS


==========================================
Testing with blocksize: 64x64
==========================================
Error: blocksize 64x64 (4096 threads) exceeds device limit of 1024 threads per block
Maximum square blocksize for this device: 32

All tests completed!
```
