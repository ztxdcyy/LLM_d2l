#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// GEMM分块参数设置
#define BM 128  // Block Tile M维度
#define BN 128  // Block Tile N维度  
#define BK 8    // Block Tile K维度
#define TM 8    // Thread Tile M维度 - 每个线程负责的行数
#define TN 8    // Thread Tile N维度 - 每个线程负责的列数

// 宏定义：计算二维数组在一维数组中的偏移
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 宏定义：FLOAT4向量化访问
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void sgemm_V1(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    // 高效的共享内存加载索引计算
    int load_a_smem_m = tid >> 1;        // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;        // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2; // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        // 向量化加载A矩阵子块到共享内存
        int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        
        // 向量化加载B矩阵子块到共享内存
        int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
        
        __syncthreads();

        // Thread Tile计算：每个线程计算TM×TN个输出元素
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

    // 向量化写回全局内存
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
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
    std::cout << "===============================" << std::endl;
    std::cout << std::endl;

    std::cout << "=== 高效Thread Tile GEMM V1 ===" << std::endl;
    std::cout << "Block Tile: BM=" << BM << ", BN=" << BN << ", BK=" << BK << std::endl;
    std::cout << "Thread Tile: TM=" << TM << ", TN=" << TN << std::endl;
    std::cout << "线程块大小: " << BN/TN << "x" << BM/TM 
              << " (" << (BN/TN) * (BM/TM) << " threads per block)" << std::endl;
    std::cout << "每个线程计算: " << TM << "x" << TN << " = " << TM*TN << " 个输出元素" << std::endl;
    std::cout << "优化特性: FLOAT4向量化加载 + 位运算索引 + Bank Conflict优化" << std::endl;
    std::cout << "共享内存使用: A[" << BM << "][" << BK << "] + B[" << BK << "][" << BN << "] = " 
              << (BM*BK + BK*BN)*sizeof(float)/1024 << " KB" << std::endl;
    std::cout << "=======================================" << std::endl;
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
        // 线程块大小：BM/TM × BN/TN = 16×16 = 256个线程
        dim3 block(BN/TN, BM/TM);  // (128/8, 128/8) = (16, 16)
        // 网格大小：按照BM和BN分块
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

        std::cout << "Matrix " << M << "x" << N
                  << " - Grid: " << grid.x << "x" << grid.y
                  << " blocks, Block: " << block.x << "x" << block.y << " threads" << std::endl;

        // 预热
        sgemm_V1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        // 执行并计时
        auto start = std::chrono::high_resolution_clock::now();
        sgemm_V1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // 检查核函数执行错误
        checkCudaError(cudaGetLastError(), "Kernel execution failed");

        // 拷贝结果回主机
        checkCudaError(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost), "Failed to copy C");

        std::cout << "高效Thread Tile GEMM (" << M << "x" << K << ") * (" << K << "x" << N
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