import threading
import time
import torch
from test_gil_effect import test_gil_effect
from test_pytorch_threading import test_pytorch_threading



def run_all_tests():
    print("🔬 Python GIL效果对比测试")
    print("=" * 50)
    
    # 测试1：纯Python计算
    print("测试1：纯Python CPU密集型计算")
    test_gil_effect()
    
    print("\n" + "=" * 50)
    
    # 测试2：PyTorch计算
    print("测试2：PyTorch矩阵计算")
    test_pytorch_threading()
    
    print("\n" + "=" * 50)
    print("📊 结论：")
    print("- 纯Python计算：受GIL限制，多线程无加速")
    print("- PyTorch计算：释放GIL，多线程有加速")
    print("- DeepSeek V2使用PyTorch操作，所以多线程是有效的")

# 将上面两个函数的代码也复制进来，然后运行：
run_all_tests()