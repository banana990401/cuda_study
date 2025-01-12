#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

// v5: multi-block reduce final result by two pass
// latency: 0.638858 ms
#define THREAD_PER_BLOCK 256

template <int blockSize>
__device__ void blockReduce(float* smem)
{
    if(blockSize >= 1024)
    {
        if(threadIdx.x < 512)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 512];
        }
        __syncthreads();
    }
    if(blockSize >= 512)
    {
        if(threadIdx.x < 256)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if(blockSize >= 256)
    {
        if(threadIdx.x < 128)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if(blockSize >= 128)
    {
        if(threadIdx.x < 64)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 64];
        }
        __syncthreads();
    }
    // the final warp
    if(threadIdx.x < 32)
    {
        volatile float* vshm = smem;

        vshm[threadIdx.x] += vshm[threadIdx.x + 32];
        vshm[threadIdx.x] += vshm[threadIdx.x + 16];
        vshm[threadIdx.x] += vshm[threadIdx.x + 8];
        vshm[threadIdx.x] += vshm[threadIdx.x + 4];
        vshm[threadIdx.x] += vshm[threadIdx.x + 2];
        vshm[threadIdx.x] += vshm[threadIdx.x + 1];
    }
}

template <int blockSize>
__global__ void reduce_v5(float* d_in, float* d_out, int nums)
{
    int tid   = threadIdx.x;
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_thread_num = blockDim.x * gridDim.x;

    __shared__ float smem[THREAD_PER_BLOCK];

    float sum = 0.0f;
    for(int i = g_tid; i < nums; i += total_thread_num)
    {
        sum += d_in[i];
    }

    smem[tid] = sum;
    __syncthreads();

    blockReduce<blockSize>(smem);

    if(tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(float* out, float groudtruth, int n)
{
    if(*out != groudtruth)
    {
        return false;
    }
    return true;
}

int main()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks       = deviceProp.maxGridSize[0];
    const int blockSize = 256;
    const int N         = 25600000;
    int gridSize        = std::min((N + blockSize - 1) / blockSize, maxblocks);

    float milliseconds = 0;
    float* a           = (float*)malloc(N * sizeof(float));
    float* d_a;
    cudaMalloc((void**)&d_a, N * sizeof(float));

    float* out = (float*)malloc((gridSize) * sizeof(float));
    float* d_out;
    float* part_out; //新增part_out存储每个block reduce的结果
    cudaMalloc((void**)&d_out, 1 * sizeof(float));
    cudaMalloc((void**)&part_out, (gridSize) * sizeof(float));
    float groudtruth = N;

    for(int i = 0; i < N; i++)
    {
        a[i] = 1;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    reduce_v5<blockSize><<<Grid, Block>>>(d_a, part_out, N);
    reduce_v5<blockSize><<<1, Block>>>(part_out, d_out, gridSize);

    int iter = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++)
    {
        reduce_v5<blockSize><<<Grid, Block>>>(d_a, part_out, N);
        reduce_v5<blockSize><<<1, Block>>>(part_out, d_out, gridSize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("grid: %d, block: %d \n", gridSize, blockSize);
    bool is_right = CheckResult(out, groudtruth, 1);
    if(is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        for(int i = 0; i < 1; i++)
        {
            printf("%lf ", out[i]);
        }
        printf("\n");
    }
    printf("reduce_v5 latency = %f ms\n", milliseconds / 10);

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(part_out);
    free(a);
    free(out);
}