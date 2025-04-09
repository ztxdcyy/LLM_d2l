#!/bin/bash

# 读取两个版本的时间数据到数组
naive_times=($(./gemm_naive | awk '/completed in:/ {print $(NF-1)}'))
shared_times=($(./gemm_sharedmem | awk '/completed in:/ {print $(NF-1)}'))

# 定义矩阵尺寸数组
sizes=(512 1024 2048 4096)

echo "性能对比报告:"
echo "----------------------------------------"

# 遍历每组数据
for i in {0..3}; do
    # 计算提升百分比
    improvement=$(echo "scale=2; ((${naive_times[$i]} - ${shared_times[$i]}) / ${naive_times[$i]}) * 100" | bc)
    
    # 输出结果
    echo "矩阵尺寸: ${sizes[$i]}x${sizes[$i]}"
    echo "朴素版本: ${naive_times[$i]}秒"
    echo "共享内存: ${shared_times[$i]}秒"
    echo "性能提升: ${improvement}%"
    echo "----------------------------------------"
done