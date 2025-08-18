# 进入项目目录
cd /home/student/ztx/dive/GEMM_UP/

g++ -O3 -o gemm GEMM.cpp

nvcc GEMM_naive.cu -o gemm_naive -arch=sm_61

# 基本性能分析（无GUI）
nsys profile --stats=true -o gemm_naive_profile ./gemm_naive

# 启动 GUI 打开性能分析报告
nsight-sys /home/student/ztx/dive/GEMM_UP/gemm_naive_profile.qdrep

