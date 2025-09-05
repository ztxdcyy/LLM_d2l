#!/usr/bin/env python3
"""
CUDA Graph 示例：展示如何通过CUDA Graph减少kernel launch开销

这个示例演示了CUDA Graph的capture和replay过程，说明为什么CUDA Graph能够显著减少kernel launch开销。
"""

import torch
import time

def simple_kernel_operations():
    """
    简单的kernel操作序列，模拟典型的深度学习计算模式
    """
    # 创建一些测试数据
    x = torch.randn(10000, 10000, device='cuda')
    y = torch.randn(10000, 10000, device='cuda')
    
    # 一系列的计算操作（每个操作都需要单独的kernel launch）
    z1 = x + y          # kernel launch 1: 加法
    z2 = z1 * 2.0       # kernel launch 2: 乘法  
    z3 = torch.relu(z2) # kernel launch 3: ReLU激活
    z4 = z3.sum()       # kernel launch 4: 求和
    
    return z4

def benchmark_normal_execution():
    """
    基准测试：正常的kernel执行方式（每次都需要kernel launch）
    """
    print("=== 正常执行模式基准测试 ===")
    
    # 预热
    for _ in range(3):
        simple_kernel_operations()
    
    # 计时
    start_time = time.time()
    for i in range(100):
        result = simple_kernel_operations()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"正常执行模式平均耗时: {avg_time * 1000:.3f} ms")
    print(f"总耗时: {(end_time - start_time) * 1000:.3f} ms")
    print(f"最终结果: {result.item():.3f}")
    
    return avg_time

def benchmark_cuda_graph_execution():
    """
    基准测试：使用CUDA Graph的执行方式（capture一次，多次replay）
    """
    print("\n=== CUDA Graph执行模式基准测试 ===")
    
    # 创建CUDA Graph对象
    graph = torch.cuda.CUDAGraph()
    
    # 预热
    for _ in range(3):
        simple_kernel_operations()
    
    # 创建静态输入tensor（CUDA Graph要求输入在capture和replay时形状不变）
    static_x = torch.randn(10000, 10000, device='cuda')
    static_y = torch.randn(10000, 10000, device='cuda')
    
    # Capture阶段：记录kernel执行序列
    print("开始CUDA Graph capture...")
    with torch.cuda.graph(graph):
        # 在graph上下文中执行操作，这些操作会被记录而不是立即执行
        static_z1 = static_x + static_y
        static_z2 = static_z1 * 2.0
        static_z3 = torch.relu(static_z2)
        static_result = static_z3.sum()
    print("CUDA Graph capture完成")
    
    # Replay阶段：多次重放captured graph
    start_time = time.time()
    for i in range(100):
        graph.replay()  # 重放整个kernel序列，只需要一次API调用
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"CUDA Graph执行模式平均耗时: {avg_time * 1000:.3f} ms")
    print(f"总耗时: {(end_time - start_time) * 1000:.3f} ms")
    print(f"最终结果: {static_result.item():.3f}")
    
    return avg_time

def explain_benefits():
    """
    解释CUDA Graph的优势
    """
    print("\n" + "="*60)
    print("CUDA Graph 优势分析:")
    print("="*60)
    print("1. 减少Kernel Launch开销:")
    print("   - 正常模式: 每个操作都需要单独的kernel launch (CPU->GPU通信)")
    print("   - Graph模式: 所有操作被capture成一个graph，只需要一次replay调用")
    print()
    print("2. 减少CPU开销:")
    print("   - 正常模式: 每次执行都需要CPU调度每个kernel")
    print("   - Graph模式: CPU只需要调用一次replay，GPU自主执行整个graph")
    print()
    print("3. 减少同步开销:")
    print("   - 正常模式: 每个kernel执行后可能需要同步")
    print("   - Graph模式: 整个graph执行完成后再同步一次")
    print()
    print("4. 适用场景:")
    print("   - 计算模式固定的推理阶段")
    print("   - 需要频繁执行相同kernel序列的场景")
    print("   - 对延迟敏感的应用")

if __name__ == "__main__":
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，请使用支持CUDA的环境运行此示例")
        exit(1)
    
    print("CUDA Graph示例 - 展示kernel launch优化")
    print("设备:", torch.cuda.get_device_name(0))
    print()
    
    try:
        # 运行基准测试
        normal_time = benchmark_normal_execution()
        graph_time = benchmark_cuda_graph_execution()
        
        # 计算加速比
        speedup = normal_time / graph_time
        print(f"\n=== 性能对比 ===")
        print(f"正常模式: {normal_time * 1000:.3f} ms/次")
        print(f"Graph模式: {graph_time * 1000:.3f} ms/次")
        print(f"加速比: {speedup:.2f}x")
        
        # 解释优势
        explain_benefits()
        
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()