//核函数的具体实现
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
// GEMM分块参数设置，按照标准GEMM优化参数
#define BM 128  // A矩阵的行分块大小
#define BN 128  // B矩阵的列分块大小
#define BK 8    // K维度的分块大小

__global__ void matmul_ShareMemory(float *A, float *B, float *C, int M, int N, int K){
    // 共享内存：存储A和B的子块
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    // 线程块和线程索引
    int bx = blockIdx.x;  // B矩阵的列块索引
    int by = blockIdx.y;  // A矩阵的行块索引
    int tx = threadIdx.x; // 线程在块内的列索引
    int ty = threadIdx.y; // 线程在块内的行索引
    
    // 计算线程负责的全局位置
    int globalRow = by * BM + ty;
    int globalCol = bx * BN + tx;
    
    // 累加结果
    float Cvalue = 0.0f;
    
    // 分块计算：遍历K维度
    for(int bk = 0; bk < (K + BK - 1) / BK; bk++){
        // 加载A矩阵子块到共享内存 (每个线程加载一个元素)
        if(ty < BM && tx < BK && globalRow < M && bk * BK + tx < K){
            As[ty][tx] = A[globalRow * K + bk * BK + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // 加载B矩阵子块到共享内存 (每个线程加载一个元素)
        if(ty < BK && tx < BN && bk * BK + ty < K && globalCol < N){
            Bs[ty][tx] = B[(bk * BK + ty) * N + globalCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads(); // 确保所有线程完成共享内存加载
        
        // 计算当前子块的矩阵乘法
        for(int k = 0; k < BK; k++){
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads(); // 确保计算完成后再进行下一轮
    }
    
    // 将结果写回全局内存
    if(globalRow < M && globalCol < N){
        C[globalRow * N + globalCol] = Cvalue;
    }
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
