#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// shared mem tile size
#define BM 128  
#define BN 128  
#define BK 8    
// TM TN 用于sharedmem内部小迭代，加载到寄存器上
#define TM 8    
#define TN 8 
// 两个很方便的宏   
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void matmul_ShareMemory(float *A, float *B, float *C, int M, int N, int K){
    // 共享内存：存储A和B的子块
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];
    
    // 线程块和线程索引，注意cuda里的索引是左上角为原点，x轴向右，y轴向下
    int bx = blockIdx.x;                // block的列索引
    int by = blockIdx.y;                // block的行索引
    int tx = threadIdx.x;               // thread在block内的列索引
    int ty = threadIdx.y;               // thread在block内的行索引 
    int tid = ty * blockDim.x + tx;     // 计算线程在block内的全局索引
    int globalRow = OFFSET(by, ty, BM);
    int globalCol = OFFSET(bx, tx, BN);
    
    // 这里仍然使用累加结果，只优化float4加载，对比下能提升多少
    float Cvalue = 0.0f;

    // 这里直接看文章中的图片，索引处。
    // 这里通过位运算，静态的加载线程，是一种只适合sharedmem tile (128,8)(8,128) 且FLOAT的加载。两者都是必要的，目的是使用上float4加载且避免使用循环。 
    // https://zhuanlan.zhihu.com/p/657632577
    int load_a_smem_m = tid >> 1;  // tid/2 = row of s_a
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    // 计算线程负责的全局位置，要将内存从gmem中搬运进smem
    // a的全局行号和b的全局列号
    // 举例，当tid=32时, s_a[16][0], s_b[1][0]; 当tid=36时，s_a[16][4],s_b[1][4]
    int load_a_gmem_m = OFFSET(by, load_a_smem_m, BM);  // global row of a
    int load_b_gmem_n = OFFSET(bx, load_b_smem_n, BN);  // global col of b
    
    // 分块计算：遍历K维度，kmn循环，从A从加载列，从B中加载行，每行每列加载完了就可以丢掉了
    for(int bk = 0; bk < K / BK; bk++){
        // load_a_gmem_k: atile全局列，由于这里bk在循环，所以得在循环内单独计算
        int load_a_gmem_k = OFFSET(bk, load_a_smem_k, BK);
        // 在循环外计算出了a的全局行，在循环内计算出了a的全局列，因此得到a在全局一维矩阵中的真正index
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);         
        // 使用FLOAT4指令加载A tile 到共享内存，同时静态指定thread加载的地址，避开循环
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(A[load_a_gmem_addr]);  
        // 同理
        int load_b_gmem_k = OFFSET(bk, load_b_smem_k, BK);
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        // load B tile 
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(B[load_b_gmem_addr]);  

        __syncthreads();            // 同步，确保所有线程都加载完了自己的任务，我们的smem加载是完整的
        
        // 计算当前子块的矩阵乘法
        if(ty < BM && tx < BN){
            for(int k = 0; k < BK; k++){
                Cvalue += s_a[ty][k] * s_b[k][tx];
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
        dim3 block(BN/TN, BM/TM);  // (128/8, 128/8) = (16, 16)
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

