#include<stdio.h>
__global__ void print_cu(){
    printf("This is GPU\n");
}

// int main(){
//     print<<<1,2>>>();
//     cudaDeviceSynchronize();
// }