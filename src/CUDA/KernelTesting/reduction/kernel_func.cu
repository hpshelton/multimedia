#include "mvec.h"
#include <climits>
extern "C" void reduce(int size, mvec *d_idata, mvec *d_odata);

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T> struct SharedMemory
{
	__device__ inline operator	   T*()
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}

	__device__ inline operator const T*() const
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}
};

template <class T> __global__ void reduce3(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    if(i < n)
		sdata[tid] = g_idata[i];
	else{
		sdata[tid].x = 0;
		sdata[tid].y = 0;
		sdata[tid].diff = INT_MAX;
	}

    if (i + blockDim.x < n) 
        sdata[tid] += g_idata[i+blockDim.x];  

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void reduce(int size, mvec *d_idata, mvec *d_odata)
{
	int n = size;
	int maxThreads = 512;
	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(mvec) : threads * sizeof(mvec);
	reduce3<mvec><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
}
