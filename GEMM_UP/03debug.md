```
    // 这里仍然使用累加结果，只优化float4加载，对比下能提升多少
    float Cvalue = 0.0f;

    // 计算当前子块的矩阵乘法
        if(ty < BM && tx < BN){
            for(int k = 0; k < BK; k++){
                Cvalue += s_a[ty][k] * s_b[k][tx];
            }
        }
    
    // 将结果写回全局内存
    if(globalRow < M && globalCol < N){
        C[globalRow * N + globalCol] = Cvalue;
    }

    
```