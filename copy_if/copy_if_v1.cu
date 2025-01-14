#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

// block:0.201632 ms
// gpu
__global__ void filter_v1(int* dst, int* nres, const int* src, int n)
{
    // 计数器声明为shared memory，去计数各个block范围内大于0的数量
    __shared__ int l_n;
    int gtid             = blockIdx.x * blockDim.x + threadIdx.x;
    int total_thread_num = blockDim.x * gridDim.x;

    for(int i = gtid; i < n; i += total_thread_num)
    {
        // use first thread to zero the counter
        // 初始化只需1个线程来操作
        if(threadIdx.x == 0)
            l_n = 0;
        __syncthreads();

        int d, pos;
        // l_n表示每个block范围内大于0的数量，block内的线程都可访问
        // pos是每个线程私有的寄存器，且作为atomicAdd的返回值，表示当前线程对l_n原子加1之前的l_n，比如1
        // 2 4号线程都大于0，那么对于4号线程来说l_n = 3, pos = 2
        if(i < n && src[i] > 0)
        {
            pos = atomicAdd(&l_n, 1);
        }
        __syncthreads();

        // 每个block选出tid=0作为leader
        // leader把每个block的l_n累加到全局计数器(nres),即所有block的局部计数器做一个reduce sum
        // 注意:
        // 下下行原子加返回的l_n为全局计数器nres原子加l_n之前的nres，比如对于block1，已知原子加前，nres
        // = 2, l_n = 3，原子加后, nres = 2+3, 返回的l_n = 2
        if(threadIdx.x == 0)
            l_n = atomicAdd(nres, l_n);
        __syncthreads();

        // write & store
        if(i < n && d > 0)
        {
            // 1. pos: src[thread]>0的thread在当前block的index
            // 2. l_n: 在当前block的前面几个block的所有src>0的个数
            // 3. pos + l_n：当前thread的全局offset
            pos += l_n;
            dst[pos] = d;
        }
        __syncthreads();
    }
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
    filter_v1<<<Grid, Block>>>(dst, nres, src, N);
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