// 共享内存 + float4 加载的 SGEMM 实现（改进版 v2）
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

// Tile sizes
#define BM 128
#define BN 128
#define BK 8
// Register tile per thread
#define TM 8
#define TN 8

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// 基于 sgemm_v1 的正确实现，使用 shared memory 和 float4 加载
__global__ void sgemm_float4(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    const int M, const int N, const int K) {

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0f};

    // 计算每个线程负责加载到 shared memory 的位置（静态分配，匹配 BK=8 的 float4 加载）
    int load_a_smem_m = tid >> 1;              // 0..255（每2个线程映射到一行）
    int load_a_smem_k = (tid & 1) << 2;        // 0 或 4（float4 对齐）
    int load_b_smem_k = tid >> 5;              // 0..31
    int load_b_smem_n = (tid & 31) << 2;       // 0,4,8,...,124（float4 对齐）

    // 对应到全局内存
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // 遍历 K 维度的 tiles
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;   // A 的全局列
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);

        int load_b_gmem_k = bk * BK + load_b_smem_k;   // B 的全局行
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_smem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

        __syncthreads();

        // 计算寄存器 tile
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;        // 0..127
                    int comp_b_smem_n = tx * TN + n;        // 0..127
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

    // 写回 C，使用 float4 打包
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

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// 采样若干元素做轻量正确性校验
double sample_max_abs_error(const std::vector<float>& A,
                            const std::vector<float>& B,
                            const std::vector<float>& C,
                            int M, int N, int K) {
    const int samples = 8;
    double max_err = 0.0;
    int coords[samples][2] = {
        {0,0}, {M/7,N/5}, {M/3,N/4}, {M/2,N/2}, {M-1,N-1}, {M/5,N/7}, {M/9,N/11}, {M/13,N/3}
    };
    for (int s = 0; s < samples; ++s) {
        int m = std::min(std::max(coords[s][0], 0), M-1);
        int n = std::min(std::max(coords[s][1], 0), N-1);
        double ref = 0.0;
        for (int k = 0; k < K; ++k) {
            ref += static_cast<double>(A[OFFSET(m,k,K)]) * static_cast<double>(B[OFFSET(k,n,N)]);
        }
        double err = std::fabs(static_cast<double>(C[OFFSET(m,n,N)]) - ref);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int sizes[] = {512, 1024, 2048, 4096};

    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::cout << "=== GPU Device Information ===\n";
    std::cout << "Device name: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    std::cout << "Multiprocessor count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n\n";

    const int repeat = 10; // 用于时间均值

    for (int width : sizes) {
        const int M = width, N = width, K = width;

        // 对齐保障（本实现依赖 4-float 对齐和整 tile 尺寸）
        if ((K % 8) != 0 || (M % 128) != 0 || (N % 128) != 0) {
            std::cerr << "Skip size " << width << ": require M,N % 128 == 0 and K % 8 == 0 for float4 path.\n";
            continue;
        }

        size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
        size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
        size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
        for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;

        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        checkCuda(cudaMalloc(&d_A, sizeA), "cudaMalloc d_A");
        checkCuda(cudaMalloc(&d_B, sizeB), "cudaMalloc d_B");
        checkCuda(cudaMalloc(&d_C, sizeC), "cudaMalloc d_C");

        checkCuda(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice), "H2D A");
        checkCuda(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice), "H2D B");

        dim3 blockDim(BN / TN, BM / TM);                 // (16,16)
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        std::cout << "Matrix " << M << "x" << N
                  << " - Grid: " << gridDim.x << "x" << gridDim.y
                  << ", Block: " << blockDim.x << "x" << blockDim.y << "\n";

        // 预热一次并做轻量正确性校验
        sgemm_float4<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        checkCuda(cudaGetLastError(), "Kernel launch (warmup)");
        checkCuda(cudaDeviceSynchronize(), "Kernel sync (warmup)");

        checkCuda(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost), "D2H C warmup");
        double max_err = sample_max_abs_error(h_A, h_B, h_C, M, N, K);
        std::cout << "Sampled max abs error: " << max_err << "\n";

        // 事件计时（仅核函数时间），重复多次取均值
        cudaEvent_t start, stop;
        checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
        checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

        checkCuda(cudaEventRecord(start), "cudaEventRecord start");
        for (int r = 0; r < repeat; ++r) {
            sgemm_float4<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }
        checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        double avg_sec = (ms / 1000.0) / repeat;

        // 统一单位到 GiFLOPS（与 sgemm_v1 一致）：除以 1024^3
        double gflops_gib = (2.0 * M * N * K) / (avg_sec * 1024.0 * 1024.0 * 1024.0);

        std::cout << "Avg time: " << avg_sec << " s, Performance: " << gflops_gib << " GiFLOPS" << std::endl;
        std::cout << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    return 0;
}

