import threading
import time

def cpu_intensive_python_task(n, result_dict, thread_id):
    """纯Python CPU密集型任务 - 会被GIL限制"""
    start_time = time.time()
    total = 0
    for i in range(n):
        total += i * i
    end_time = time.time()
    result_dict[thread_id] = {
        'result': total,
        'time': end_time - start_time
    }

def test_gil_effect():
    n = 50000000  # 5千万次计算
    
    # 单线程执行
    print("=== 单线程执行（纯Python计算）===")
    start_time = time.time()
    result_dict = {}
    cpu_intensive_python_task(n, result_dict, 'single')
    single_thread_time = time.time() - start_time
    print(f"单线程时间: {single_thread_time:.2f}秒")
    
    # 多线程执行
    print("\n=== 多线程执行（纯Python计算）===")
    start_time = time.time()
    result_dict = {}
    threads = []
    
    # 创建2个线程，每个处理一半工作
    for i in range(2):
        thread = threading.Thread(
            target=cpu_intensive_python_task, 
            args=(n//2, result_dict, f'thread_{i}')
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    multi_thread_time = time.time() - start_time
    print(f"多线程时间: {multi_thread_time:.2f}秒")
    print(f"加速比: {single_thread_time/multi_thread_time:.2f}x")
    
    if multi_thread_time >= single_thread_time * 0.9:
        print("❌ GIL生效：多线程几乎没有加速效果")
    else:
        print("✅ 多线程有效果")

if __name__ == "__main__":
    test_gil_effect()