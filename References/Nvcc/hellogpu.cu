#include <stdio.h>
#include <stdlib.h>

void cpu(int *a,int N){
    for(int i=0;i<N;i++){
        a[i]=i;
    }
}
// 标志在GPU运行且是全局调用的，返回值必须是void
__global__ void gpu(int *a, int N){
    int threadi =blockIdx.x*blockDim.x+threadIdx.x;
    int stride =gridDim.x*blockDim.x;
    for (int i=threadi;i<N;i+=stride){
        a[i]*=2;        
    }
}
bool check(int *a,int N){
    for(int i=0;i<N;i++){
        if(a[i]!=2*i) return false;
    }
    return true;
}
int main(){
    const int N=10000;
    size_t size=N*sizeof(int);
    int *a;
    // a=(int*)malloc(size); 
    cudaMallocManaged(&a,size); 
    cpu(a,N);
    size_t threads=256;
    size_t blocks=(N+threads-1)/threads;//上取整

    gpu<<<blocks,threads>>>(a,N);
    cudaDeviceSynchronize();
    check(a,N) ? printf("ok\n") : printf("error\n");

    cudaFree(a);
    // free(a);
    // // 表示启动函数的操作，GPU线程块和线程个数 <<<block,thread>>
    // gpu<<<2,2>>>();
    // // 同步操作，CPU等待GPU完成后才能继续执行
    // cudaDeviceSynchronize();
}