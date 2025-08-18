//核函数的具体实现 - 支持可调节blocksize的版本
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

__global__ void matMul_GlobalKernel(float *A, float *B, float *C, int width){
    // md，Idx = Index，最后一个x意思的index的x，而不是xyz的x。有blockIdx.z
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 对于 block划分成《blockDim.x, blockDim.y》， threadid可以通过下面两个公式计算出来
    int Col = bx * blockDim.x + tx;         // col 列
    int Row = by * blockDim.y + ty;         // row 行

    // 边界检查，防止越界访问
    if (Row >= width || Col >= width) return;

    float perValue = 0.0f;

    for(int i = 0; i < width; i++){
        perValue += A[Row * width + i] * B[i * width + Col];
    }
    C[Row * width + Col] = perValue;
}

// 检查CUDA错误
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [blocksize]" << std::endl;
    std::cout << "  blocksize: Size of thread block (default: 32, must be power of 2)" << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "    " << programName << " 16    # Use 16x16 thread blocks" << std::endl;
    std::cout << "    " << programName << " 32    # Use 32x32 thread blocks" << std::endl;
    std::cout << "    " << programName << " 64    # Use 64x64 thread blocks (if supported)" << std::endl;
}

int main(int argc, char* argv[]) {
    // 获取设备属性以确定最大blocksize
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxBlockSize = (int)sqrt(prop.maxThreadsPerBlock);  // 计算最大的方形blocksize
    
    // 解析命令行参数
    int blockSize = 32;  // 默认值
    if (argc > 1) {
        blockSize = std::atoi(argv[1]);
        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0) {
            std::cerr << "Error: blocksize must be a power of 2" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        if (blockSize * blockSize > prop.maxThreadsPerBlock) {
            std::cerr << "Error: blocksize " << blockSize << "x" << blockSize 
                      << " (" << blockSize * blockSize << " threads) exceeds device limit of " 
                      << prop.maxThreadsPerBlock << " threads per block" << std::endl;
            std::cerr << "Maximum square blocksize for this device: " << maxBlockSize << std::endl;
            return 1;
        }
    }

    const int sizes[] = {512, 1024, 2048, 4096};

    std::cout << "=== GPU Device Information ===" << std::endl;
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max grid size: " 
              << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x "
              << prop.maxGridSize[2] << std::endl;
    std::cout << "Max block dimensions: " 
              << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << std::endl;

    std::cout << "Using block size: " << blockSize << "x" << blockSize 
              << " (" << blockSize * blockSize << " threads per block)" << std::endl;
    std::cout << "Maximum supported square blocksize: " << maxBlockSize << std::endl;
    std::cout << std::endl;

    for (int width : sizes) {
        // 分配主机内存
        size_t matrixSize = width * width * sizeof(float);
        float *h_A = new float[width * width];
        float *h_B = new float[width * width];
        float *h_C = new float[width * width];

        // 初始化随机数
        std::srand(std::time(nullptr));
        for (int i = 0; i < width * width; i++) {
            h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        // 分配设备内存
        float *d_A, *d_B, *d_C;
        checkCudaError(cudaMalloc(&d_A, matrixSize), "Failed to allocate d_A");
        checkCudaError(cudaMalloc(&d_B, matrixSize), "Failed to allocate d_B");
        checkCudaError(cudaMalloc(&d_C, matrixSize), "Failed to allocate d_C");

        // 拷贝数据到设备
        checkCudaError(cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice), "Failed to copy A");
        checkCudaError(cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice), "Failed to copy B");

        // 配置CUDA核函数，按照blocksize划分grid，使得整个输入矩阵width都能被per element的分配到内存上。
        // dim3 block(blockSize, blockSize);
        // dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
        dim3 block(64, 64);
        dim3 grid((width + 63) / 64, (width + 63) / 64);

        std::cout << "Matrix " << width << "x" << width 
                  << " - Grid: " << grid.x << "x" << grid.y 
                  << " blocks, Block: " << block.x << "x" << block.y << " threads" << std::endl;

        // 执行并计时
        auto start = std::chrono::high_resolution_clock::now();
        matMul_GlobalKernel<<<grid, block>>>(d_A, d_B, d_C, width);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // 检查核函数执行错误
        checkCudaError(cudaGetLastError(), "Kernel execution failed");

        // 拷贝结果回主机
        checkCudaError(cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost), "Failed to copy C");

        std::cout << "Matrix multiplication (" << width << "x" << width << ") completed in: " 
                  << elapsed.count() << " seconds" << std::endl;
        
        // 计算GFLOPS
        double gflops = (2.0 * width * width * width) / (elapsed.count() * 1e9);
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
        std::cout << std::endl;

        // 释放资源
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }
    return 0;
}