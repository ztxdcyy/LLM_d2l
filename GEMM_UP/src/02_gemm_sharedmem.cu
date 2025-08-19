//核函数的具体实现 - 共享内存优化版本
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// GEMM分块参数设置，按照标准GEMM优化参数
#define BM 128  // A矩阵的行分块大小
#define BN 128  // B矩阵的列分块大小
#define BK 8    // K维度的分块大小
#define RM 8    // 每个线程负责的行数
#define RN 8    // 每个线程负责的列数

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
    for(int bk = 0; bk < K / BK; bk++){
        // 加载A矩阵子块到共享内存
        // 每个线程负责加载多个元素以填满共享内存
        for(int loadRow = ty; loadRow < BM; loadRow += blockDim.y){
            for(int loadCol = tx; loadCol < BK; loadCol += blockDim.x){
                int globalLoadRow = by * BM + loadRow;
                int globalLoadCol = bk * BK + loadCol;
                if(globalLoadRow < M && globalLoadCol < K){
                    As[loadRow][loadCol] = A[globalLoadRow * K + globalLoadCol];            // A：行优先存储在一维数组中
                } else {
                    As[loadRow][loadCol] = 0.0f;
                }
            }
        }

        // 加载B矩阵子块到共享内存
        for(int loadRow = ty; loadRow < BK; loadRow += blockDim.y){
            for(int loadCol = tx; loadCol < BN; loadCol += blockDim.x){
                int globalLoadRow = bk * BK + loadRow;
                int globalLoadCol = bx * BN + loadCol;
                if(globalLoadRow < K && globalLoadCol < N){
                    Bs[loadRow][loadCol] = B[globalLoadRow * N + globalLoadCol];
                } else {
                    Bs[loadRow][loadCol] = 0.0f;
                }
            }
        }
        
        __syncthreads(); // 确保所有线程完成共享内存加载
        
        // 计算当前子块的矩阵乘法
        if(ty < BM && tx < BN){
            for(int k = 0; k < BK; k++){
                Cvalue += As[ty][k] * Bs[k][tx];
            }
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
    // 初始化随机数种子
    std::srand(std::time(nullptr));
    
    const int sizes[] = {512, 1024, 2048, 4096};

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "=== GPU Device Information ===" << std::endl;
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;

    for (int width : sizes) {
        int M = width, N = width, K = width;  // 方阵
        // 分配主机内存
        float *h_A = new float[M * K];
        float *h_B = new float[K * N];
        float *h_C = new float[M * N];

        // 初始化矩阵
        for (int i = 0; i < M * K; i++) {
            h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }
        for (int i = 0; i < K * N; i++) {
            h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        // 分配设备内存
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);
        
        float *d_A, *d_B, *d_C;
        checkCudaError(cudaMalloc(&d_A, sizeA), "Failed to allocate d_A");
        checkCudaError(cudaMalloc(&d_B, sizeB), "Failed to allocate d_B");
        checkCudaError(cudaMalloc(&d_C, sizeC), "Failed to allocate d_C");

        // 拷贝数据到设备
        checkCudaError(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice), "Failed to copy A");
        checkCudaError(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice), "Failed to copy B");

        // 配置CUDA核函数
        // 线程块大小：根据BM/RM和BN/RN计算
        dim3 block(BN/RN, BM/RM);  // (128/8, 128/8) = (16, 16)
        // 网格大小：按照BM和BN分块大小计算
        dim3 grid(N / BN, M / BM);
        
        std::cout << "Matrix " << M << "x" << N
                  << " - Grid: " << grid.x << "x" << grid.y
                  << " blocks, Block: " << block.x << "x" << block.y << " threads" << std::endl;

        // 执行并计时
        auto start = std::chrono::high_resolution_clock::now();
        matmul_ShareMemory<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // 检查核函数执行错误
        checkCudaError(cudaGetLastError(), "Kernel execution failed");

        // 拷贝结果回主机
        checkCudaError(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost), "Failed to copy C");

        std::cout << "Matrix multiplication (" << M << "x" << K << ") * (" << K << "x" << N
                  << ") completed in: " << elapsed.count() << " seconds" << std::endl;
        
        // 计算GFLOPS
        double gflops = (2.0 * M * N * K) / (elapsed.count() * 1e9);
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
