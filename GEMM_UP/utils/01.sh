#!/bin/bash

# 编译程序
echo "Compiling src/01_gemm_blocksize.cu..."
nvcc -o bin/01_gemm_blocksize src/01_gemm_blocksize.cu

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# 测试不同的 blocksize
blocksizes=(8 16 32 64)

for blocksize in "${blocksizes[@]}"; do
    echo "=========================================="
    echo "Testing with blocksize: ${blocksize}x${blocksize}"
    echo "=========================================="
    ./bin/01_gemm_blocksize $blocksize
    echo ""
done

echo "All tests completed!"