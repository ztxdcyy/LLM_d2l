#!/bin/bash

echo "=== GEMM共享内存版本编译和运行脚本 ==="
echo "项目根目录: $(pwd)"
echo ""

echo "=== 开始编译 ==="
# 编译程序，使用sm_90架构
NVCC_FLAGS="-O3 -arch=sm_90 -std=c++11"
nvcc $NVCC_FLAGS -o bin/02_gemm_sharedmem src/02_gemm_sharedmem.cu

if [ $? -ne 0 ]; then
    echo "❌ 编译失败!"
    exit 1
fi

echo "✅ 编译成功!"
echo ""

echo "=== 开始运行 ==="
echo "执行文件: $(pwd)/bin/02_gemm_sharedmem"
echo ""

# 运行程序
./bin/02_gemm_sharedmem

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 程序运行完成!"
else
    echo ""
    echo "❌ 程序运行失败!"
fi

echo ""
echo "=== 脚本执行完成 ==="