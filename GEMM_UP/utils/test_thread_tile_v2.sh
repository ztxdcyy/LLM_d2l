#!/bin/bash

# 高效Thread Tile GEMM V2测试脚本
echo "=== 编译和测试高效Thread Tile GEMM V2 ==="

# 设置编译参数
NVCC_FLAGS="-O3 -arch=sm_75 -std=c++11"
SOURCE_FILE="src/03_gemm_thread_tile_v2.cu"
OUTPUT_FILE="bin/03_gemm_thread_tile_v2"

# 创建bin目录
mkdir -p bin

# 编译
echo "正在编译 $SOURCE_FILE ..."
nvcc $NVCC_FLAGS $SOURCE_FILE -o $OUTPUT_FILE

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo ""
    
    # 运行测试
    echo "=== 运行高效Thread Tile GEMM V2测试 ==="
    echo "优化特性："
    echo "- FLOAT4向量化加载/存储"
    echo "- 位运算索引计算优化"
    echo "- 共享内存Bank Conflict避免"
    echo "- Thread Tile: 每个线程计算8x8=64个输出元素"
    echo ""
    
    ./$OUTPUT_FILE
    
    echo ""
    echo "=== 测试完成 ==="
    echo "如需与原版本对比，请运行："
    echo "  ./utils/test_sharedmem.sh"
    echo "  ./utils/test_thread_tile.sh"
    
else
    echo "编译失败！"
    exit 1
fi