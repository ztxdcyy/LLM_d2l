参考：
https://zhuanlan.zhihu.com/p/1902778199261291694

```
(base) root@autodl-container-10be4ca793-400c39a3:~/workspace/LLM_d2l/Triton# python 01_triton_add.py 
PyTorch add elapsed: 24.128448 ms
Triton add elapsed: 550.181946 ms
```

Triton内核第一次启动有编译和JIT开销，但代码只运行一次：
```
# 当前代码只运行一次，无法消除JIT编译开销
output_triton = add(x, y)
```