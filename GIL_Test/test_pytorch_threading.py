import threading
import time
import torch

def pytorch_computation_task(size, result_dict, thread_id):
    """PyTorch计算任务 - 会释放GIL"""
    start_time = time.time()
    
    # 创建大矩阵并进行计算
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Thread {thread_id}: 使用GPU计算")
    else:
        device = torch.device('cpu')
        print(f"Thread {thread_id}: 使用CPU计算")
    
    # 矩阵乘法计算
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 多次矩阵乘法
    for _ in range(10):
        c = torch.mm(a, b)
        a = c
    
    end_time = time.time()
    result_dict[thread_id] = {
        'result': c.sum().item(),
        'time': end_time - start_time
    }

def test_pytorch_threading():
    size = 2000  # 2000x2000矩阵
    
    # 单线程执行
    print("=== 单线程执行（PyTorch计算）===")
    start_time = time.time()
    result_dict = {}
    pytorch_computation_task(size, result_dict, 'single')
    single_thread_time = time.time() - start_time
    print(f"单线程时间: {single_thread_time:.2f}秒")
    
    # 多线程执行
    print("\n=== 多线程执行（PyTorch计算）===")
    start_time = time.time()
    result_dict = {}
    threads = []
    
    # 创建2个线程
    for i in range(2):
        thread = threading.Thread(
            target=pytorch_computation_task, 
            args=(size//2, result_dict, f'thread_{i}')
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    multi_thread_time = time.time() - start_time
    print(f"多线程时间: {multi_thread_time:.2f}秒")
    print(f"加速比: {single_thread_time/multi_thread_time:.2f}x")
    
    if multi_thread_time < single_thread_time * 0.8:
        print("✅ PyTorch释放GIL：多线程有明显加速效果")
    else:
        print("⚠️ 加速效果不明显（可能受硬件限制）")

if __name__ == "__main__":
    test_pytorch_threading()