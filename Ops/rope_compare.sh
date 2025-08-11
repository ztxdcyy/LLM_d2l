#!/bin/bash

echo "性能对比测试开始..."
echo "-----------------------------"

# 测试参数设置
BATCH_SIZE=3
NUM_HEADS=16
HEAD_DIM=128
NUM_RUNS=10
SEQ_LEN_LIST=(256 512 1024)

for SEQ_LEN in "${SEQ_LEN_LIST[@]}"; do
    echo "测试序列长度: $SEQ_LEN"
    echo "-----------------------------"
    
    # 测试RoPE1
    echo "测试RoPE1实现..."
    rope1_times=()
    for ((i=1; i<=NUM_RUNS; i++)); do
        elapsed=$(python RoPE1.py $BATCH_SIZE $SEQ_LEN $NUM_HEADS $HEAD_DIM)
        rope1_times+=($elapsed)
    done
    rope1_avg=$(printf "%s\n" "${rope1_times[@]}" | awk '{sum+=$1} END {print sum/NR}')
    echo "RoPE1 平均耗时: ${rope1_avg}秒"

    # 测试RoPE2
    echo "测试RoPE2实现..."
    rope2_times=()
    for ((i=1; i<=NUM_RUNS; i++)); do
        elapsed=$(python RoPE2.py $BATCH_SIZE $SEQ_LEN $NUM_HEADS $HEAD_DIM)
        rope2_times+=($elapsed)
    done
    rope2_avg=$(printf "%s\n" "${rope2_times[@]}" | awk '{sum+=$1} END {print sum/NR}')
    echo "RoPE2 平均耗时: ${rope2_avg}秒"
    
    # 性能对比结果
    echo "-----------------------------"
    echo "性能对比结果(seq_len=$SEQ_LEN):"
    speedup=$(awk -v r2="$rope2_avg" -v r1="$rope1_avg" 'BEGIN {printf "%.2f", r2/r1}')
    echo "RoPE1 相对于 RoPE2 的加速比: ${speedup}x"
    echo "================================="
done