#include <stdio.h>
#include <cuda_runtime.h>

A = [1, K]
B = [K, N]
C = A*B (1,N)

__global__ void gemv(
    float* A, 
    float* B, 
    float* C, 
    int K, int N)
{
    int j = block


}