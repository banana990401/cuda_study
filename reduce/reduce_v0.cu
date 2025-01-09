#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

// v0: naive版本
// latency: 0.959590 ms
// blockSize作为模板参数的效果主要用于静态shared memory的申请需要传入编译期常量指定大小（L10)
template <int blockSize>
__global__ void reduce_v0(float* d_in, float* d_out)
{
    int tid   = threadIdx.x;
    int g_tid = blockIdx.x * blockSize + threadIdx.x;

    __shared__ float smem[blockSize];

    smem[tid] = d_in[g_tid];
    __syncthreads();

    for(int stride = 1; stride < blockSize; stride *= 2)
    {
        if((tid & (2 * stride - 1)) == 0)
        {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

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
    dim3 Block(blockSize);

    // warm up
    reduce_v0<blockSize><<<Grid, Block>>>(d_a, d_out);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int iter = 10;
    for(int i = 0; i < iter; i++)
    {
        reduce_v0<blockSize><<<Grid, Block>>>(d_a, d_out);
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
    printf("reduce_v0 latency = %f ms\n", milliseconds / iter);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
