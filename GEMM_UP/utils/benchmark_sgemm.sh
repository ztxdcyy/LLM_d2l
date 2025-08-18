#!/bin/bash

# 专业GEMM基准测试脚本
echo "=== 编译和运行专业GEMM基准测试 ==="

# 设置编译参数
NVCC_FLAGS="-O3 -arch=sm_75 -std=c++11"
SOURCE_FILE="src/04_gemm_benchmark.cu"
OUTPUT_FILE="bin/04_gemm_benchmark"

# 创建bin目录
mkdir -p bin

# 编译
echo "正在编译 $SOURCE_FILE ..."
echo "编译参数: $NVCC_FLAGS"
echo "链接cuBLAS库进行精度验证..."

nvcc $NVCC_FLAGS $SOURCE_FILE -o $OUTPUT_FILE -lcublas

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo ""
    
    # 运行基准测试
    echo "=== 运行专业GEMM基准测试 ==="
    echo "测试特性："
    echo "- 与cuBLAS进行精度对比验证"
    echo "- 15种不同矩阵尺寸的性能测试"
    echo "- 10次重复测试取平均值"
    echo "- FLOAT4向量化 + 位运算索引优化"
    echo "- Thread Tile: 每个线程计算8x8=64个输出元素"
    echo ""
    
    ./$OUTPUT_FILE
    
    echo ""
    echo "=== 基准测试完成 ==="
    echo "性能数据已输出，可用于与其他实现对比"
    
else
    echo "编译失败！请检查cuBLAS库是否正确安装"
    exit 1
fi