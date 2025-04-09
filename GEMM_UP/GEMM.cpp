#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

void matrixMulCpu(float *A, float *B, float *C, int width)
{
    float sum = 0.0f;
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int l = 0; l < width; l++)
            {   
                // 二维矩阵在内存中是按行主序连续存储的，i * width + j来访问第i行第j列的元素，l代表当前索引
                sum += A[i * width + l] * B[l * width + j];
            }
            C[i * width + j] = sum;
            sum = 0.0f;
        }
    }
}

int main()
{
    // C风格数组，轻量，没有那么多方法
    const int sizes[] = {512, 1024, 2048};

    for (int width : sizes) {
        // 直接创建局部变量的话可能因为size过大造成栈溢出，所以使用new来分配内存
        float *A = new float[width * width];
        float *B = new float[width * width];
        float *C = new float[width * width];

        // 初始化随机数种子
        std::srand(std::time(nullptr));

        // 填充随机数到矩阵A和B
        for (int i = 0; i < width * width; i++)
        {
            // std::rand()生成的是整数数据，通过除以最大值，可以将其转换为浮点数
            // RAND_MAX是std::rand()返回的最大值
            // static_cast<float>强制转换单精度FP32，这样和后面整数相除的时候才会触发浮点除法
            A[i] = static_cast<float>(std::rand()) / RAND_MAX;
            B[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        // 开始计时
        auto start = std::chrono::high_resolution_clock::now();

        // 执行矩阵乘法
        matrixMulCpu(A, B, C, width);

        // 结束计时
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Matrix multiplication (" << width << "x" << width << ") completed in: " 
                  << elapsed.count() << " seconds" << std::endl;

        delete[] A;
        delete[] B;
        delete[] C;
    }
    return 0;
}