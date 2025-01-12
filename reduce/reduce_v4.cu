#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

// v4: 循环展开
// latency: 0.414413 ms
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
__global__ void reduce_v3(float* d_in, float* d_out)
{
    int tid   = threadIdx.x;
    int g_tid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    __shared__ float smem[THREAD_PER_BLOCK];

    smem[tid] = d_in[g_tid] + d_in[g_tid + blockDim.x];
    __syncthreads();

    blockReduce<blockSize>(smem);

    if(tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
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
    dim3 Block(blockSize / 2);

    // warm up
    reduce_v3<blockSize / 2><<<Grid, Block>>>(d_a, d_out);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int iter = 10;
    for(int i = 0; i < iter; i++)
    {
        reduce_v3<blockSize / 2><<<Grid, Block>>>(d_a, d_out);
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
        // for(int i = 0; i < GridSize;i++){
        // printf("res per block : %lf ",out[i]);
        //}
        // printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v3 latency = %f ms\n", milliseconds / iter);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
