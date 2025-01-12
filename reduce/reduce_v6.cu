// warp shuffle
#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define WARP_SIZE 32
// v6：warp reduce
//latency: 0.505549 ms
__device__ float WarpShuffle(float sum)
{
    // __shfl_down_sync：前面的thread向后面的thread要数据
    // __shfl_up_sync: 后面的thread向前面的thread要数据
    // 1. 返回前面的thread向后面的thread要的数据，比如__shfl_down_sync(0xffffffff, sum,
    // 16)那就是返回16号线程，17号线程的数据
    // 2. 使用warp shuffle指令的数据交换不会出现warp在shared
    // memory上交换数据时的不一致现象，这一点是由GPU driver完成，故无需任何sync, 比如syncwarp
    // 3. 原先15-19行有5个if判断block size的大小，目前已经被移除，确认了一下__shfl_down_sync等warp
    // shuffle指令可以handle一个block或一个warp的线程数量<32，不足32会自动填充0
    sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
    return sum;
}

template <int blockSize>
__global__ void reduce_v6(float* d_in, float* d_out, unsigned int n)
{
    float sum = 0; //当前线程的私有寄存器，即每个线程都会拥有一个sum寄存器

    unsigned int tid  = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_thread_num = blockDim.x * gridDim.x;
    for(int i = gtid; i < n; i += total_thread_num)
    {
        sum += d_in[i];
    }

    // 用于存储partial sums for each warp of a block
    __shared__ float warp_sums[blockSize / WARP_SIZE];
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    // 对当前线程所在warp作warpshuffle操作，直接交换warp内线程间的寄存器数据
    sum = WarpShuffle(sum);
    if(lane_id == 0)
    {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    //至此，得到了每个warp的reduce sum结果
    //接下来，再使用第一个warp(lane_id=0-31)对每个warp的reduce sum结果求和
    //首先，把warp_sums存入前blockDim.x / WARP_SIZE个线程的sum寄存器中
    //接着，继续warpshuffle
    sum = (tid < blockSize / WARP_SIZE) ? warp_sums[lane_id] : 0;
    // Final reduce using first warp
    if(warp_id == 0)
    {
        sum = WarpShuffle(sum);
    }
    if(tid == 0)
    {
        d_out[blockIdx.x] = sum;
    }
}

bool CheckResult(float* out, float groudtruth, int n)
{
    float res = 0;
    for(int i = 0; i < n; i++)
    {
        res += out[i];
    }
    if(res != groudtruth)
    {
        return false;
    }
    return true;
}

int main()
{
    float milliseconds = 0;
    const int N        = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize        = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    // int GridSize = 100000;
    float* a = (float*)malloc(N * sizeof(float));
    float* d_a;
    cudaMalloc((void**)&d_a, N * sizeof(float));

    float* out = (float*)malloc((GridSize) * sizeof(float));
    float* d_out;
    cudaMalloc((void**)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    reduce_v6<blockSize><<<Grid, Block>>>(d_a, d_out, N);

    int iter = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++)
    {
        reduce_v6<blockSize><<<Grid, Block>>>(d_a, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d \n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        for(int i = 0; i < GridSize; i++)
        {
            printf("resPerBlock : %lf ", out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v6 latency = %f ms\n", milliseconds / iter);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
