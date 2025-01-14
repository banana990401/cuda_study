#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

// warp level, use warp-aggregated atomics
__device__ int atomicAggInc(int* ctr)
{
    // 获取当前 Warp 中活跃线程的掩码
    unsigned int active = __activemask();
    // 找到当前 Warp 中的 leader 线程（第一个活跃线程）
    int leader = __ffs(active) - 1;
    // 计算当前 Warp 中活跃线程的数量
    int change = __popc(active);
    // 获取当前线程的 lane mask less than（表示比当前线程 ID 小的线程掩码）
    int lane_mask_lt;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
    // 计算当前线程在 Warp 中的局部偏移量（rank）
    unsigned int rank = __popc(active & lane_mask_lt);
    // Warp 级别的全局偏移量
    int warp_res;
    // 只有 leader 线程执行全局原子操作
    if(rank == 0)
        warp_res = atomicAdd(ctr, change); // 计算当前 Warp 的全局偏移量
    // 将 leader 线程的 warp_res 广播到所有活跃线程
    warp_res = __shfl_sync(active, warp_res, leader);
    // 返回当前线程的最终偏移量（全局偏移量 + 局部偏移量）
    return warp_res + rank;
}

// warp:0.168576 ms
// gpu
__global__ void filter_v2(int* dst, int* nres, const int* src, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= n)
        return;
    if(src[i] >
       0) // 过滤出src[i] > 0
          // 的线程，比如warp0里面只有0号和1号线程的src[i]>0，那么只有0号和1号线程运行L91，对应的L72的__activemask()为110000...00
        // 以上L71函数计算当前thread负责数据的全局offset
        dst[atomicAggInc(nres)] = src[i];
}

// cpu
int filter(int* dst, const int* src, int n)
{
    int nres = 0;
    for(int i = 0; i < n; i++)
    {
        if(src[i] > 0)
        {
            dst[nres++] = src[i];
        }
    }
    return nres;
}

bool CheckResult(int* out, int groudtruth, int n)
{
    if(*out != groudtruth)
    {
        return false;
    }
    return true;
}

int main()
{
    float milliseconds = 0;
    int N              = 2560000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize        = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

    int* src_h  = (int*)malloc(N * sizeof(int));
    int* dst_h  = (int*)malloc(N * sizeof(int));
    int* nres_h = (int*)malloc(1 * sizeof(int));
    int *dst, *nres;
    int* src;
    cudaMalloc((void**)&src, N * sizeof(int));
    cudaMalloc((void**)&dst, N * sizeof(int));
    cudaMalloc((void**)&nres, 1 * sizeof(int));

    for(int i = 0; i < N; i++)
    {
        src_h[i] = 1;
    }

    int groudtruth = 0;
    for(int j = 0; j < N; j++)
    {
        if(src_h[j] > 0)
        {
            groudtruth += 1;
        }
    }

    cudaMemcpy(src, src_h, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    filter_v2<<<Grid, Block>>>(dst, nres, src, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(nres_h, groudtruth, N);
    if(is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        printf("%d ", *nres_h);
        printf("\n");
    }
    printf("filter_k latency = %f ms\n", milliseconds);

    cudaFree(src);
    cudaFree(dst);
    cudaFree(nres);
    free(src_h);
    free(dst_h);
    free(nres_h);
}