#!/bin/bash

echo "=== GEMM共享内存版本编译和运行脚本 ==="
echo "项目根目录: $(pwd)"
echo ""

echo "=== 开始编译 ==="
# 编译程序，选择使用的架构
NVCC_FLAGS="-O3 -arch=sm_89 -std=c++11"
nvcc $NVCC_FLAGS -o bin/03 src/03_gemm_sharedmem_FLOAT4load.cu

echo "=== 开始运行 ==="
echo "执行文件: $(pwd)/bin/03"
echo ""

# 运行程序
./bin/03

echo ""
echo "=== 脚本执行完成 ==="