//核函数的具体实现
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

__global__ void matmul_ShareMemory(float *M,float *N,float *P,int width){
    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = bx * BLOCK_SIZE + tx;
    int Row = by * BLOCK_SIZE + ty;

    int Pervalue = 0;
    //有多少个BLOCK_SIZE，每个循环计算一个块的大小
    for(int i = 0;i < width / BLOCK_SIZE;i++){
        Mds[ty][tx] = M[Row * width + (i * BLOCK_SIZE + tx)];
        Nds[ty][tx] = N[Col + (i * BLOCK_SIZE + ty) * width];
        __syncthreads();        // 确保所有线程都完成了共享内存的写入

        //BLOCK_SIZE相乘
        for(int k = 0;k < BLOCK_SIZE;k++)
            Pervalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    P[Row * width + Col] = Pervalue;
}

// 检查CUDA错误
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int sizes[] = {512, 1024, 2048, 4096};
    // const int blockSize = 32;               // 为了满足研究warp的性质，设置blocksize为32，因为warp的定义是32个内存连续的线程，统一定义为一个事件，接受指令transaction，而不是一个线程一个事件，这样可以减少指令的数量，提高效率。

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Max grid size: " 
            << prop.maxGridSize[0] << " x "
            << prop.maxGridSize[1] << " x "
            << prop.maxGridSize[2] << std::endl;
    std::cout << "Max block dimensions: " 
            << prop.maxThreadsDim[0] << " x "
            << prop.maxThreadsDim[1] << " x "
            << prop.maxThreadsDim[2] << std::endl;

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
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

        // 执行并计时
        auto start = std::chrono::high_resolution_clock::now();
        matmul_ShareMemory<<<grid, block>>>(d_A, d_B, d_C, width);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // 检查核函数执行错误
        checkCudaError(cudaGetLastError(), "Kernel execution failed");

        // 拷贝结果回主机
        checkCudaError(cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost), "Failed to copy C");

        std::cout << "Matrix multiplication (" << width << "x" << width << ") completed in: " << elapsed.count() << " seconds" << std::endl;

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
