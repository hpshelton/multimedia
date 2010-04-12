#include <cutil_inline.h>
#include "kernels.cu"

extern "C" void edgeDetectGPU(unsigned char* input, unsigned char* output, int row, int col);
extern "C" void CUfwt97   (float* output, unsigned char* input, float* tempbank, int n);
extern "C" void CUfwt97_2D(float* output, unsigned char* input, float* tempbank, int row, int col);
extern "C" void CUiwt97   (unsigned char* output, float* input, float* tempbank, int n);
extern "C" void CUiwt97_2D(unsigned char* output, float* input, float* tempbank, int row, int col);
extern "C" void CUquantize(float* x, int Qlevel, int maxval, int len);
extern "C" void CUsetToVal(unsigned char* x, int len, int val);

void CUquantize(float* x, int Qlevel, int maxval, int len)
{
	int threadsPerBlock = 512;
	int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
	quantize<<<blocksPerGrid, threadsPerBlock>>>(x, Qlevel, maxval, len);
}

void CUtranspose(float* d_odata, float* d_idata, int col, int row)
{
	dim3 grid(col / BLOCK_DIM, row / BLOCK_DIM, 1);
	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
	transpose<<< grid, threads >>>(d_odata, d_idata, col, row);
}

void CUsetToVal(unsigned char* x, int len, int val)
{
	int threadsPerBlock = len;
	int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
	setToVal<<<blocksPerGrid, threadsPerBlock>>>(x, len, val);
}

void edgeDetectGPU(unsigned char* input, unsigned char* output, int row, int col)
{
	edge_detect<<<row,col>>>(input, output, row, col);
}

/*
	n is the length of input, which must be a power of 2
	output and tempbank should also be of length n
*/
void CUfwt97(float* output, unsigned char* input, float* tempbank, int n)
{
	int threadsPerBlock = n;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	// execute the kernel
	fwt97<<<blocksPerGrid, threadsPerBlock>>>(output, input, tempbank, n);
}

void CUfwt97_2D(float* output, unsigned char* input, float* tempbank, int row, int col)
{
	int i;
	int threadsPerBlock = col;
	int blocksPerGrid = (col + threadsPerBlock - 1) / threadsPerBlock;
	float* outputT;
	cutilSafeCall(cudaMalloc((void**)&outputT, sizeof(float)*row*col));

	// execute the kernel
	for(i=0; i < row; i++)
		fwt97<<<blocksPerGrid, threadsPerBlock>>>(&outputT[i*col], &input[i*col], tempbank, col);
	CUtranspose(output, outputT, col, row);
	for(i=0; i < col; i++)
		fwt97<<<blocksPerGrid, threadsPerBlock>>>(&output[i*row], tempbank, row);

	cutilSafeCall(cudaFree(outputT));
}

/*
	n is the length of input, which must be a power of 2
	output and tempbank should also be of length n
*/
void CUiwt97(unsigned char* output, float* input, float* tempbank, int n)
{
	int threadsPerBlock = n;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	// execute the kernel
	iwt97<<<blocksPerGrid, threadsPerBlock>>>(output, input, tempbank, n);
}

void CUiwt97_2D(unsigned char* output, float* input, float* tempbank, int row, int col)
{
	int i;
	int threadsPerBlock = col;
	int blocksPerGrid = (col + threadsPerBlock - 1) / threadsPerBlock;
	float* inputT;
	cutilSafeCall(cudaMalloc((void**)&inputT, sizeof(float)*row*col));

	// execute the kernel

	for(i=0; i < col; i++)
		iwt97<<<blocksPerGrid, threadsPerBlock>>>(&input[i*row], tempbank, row);
	CUtranspose(inputT, input, row, col);
	for(i=0; i < row; i++)
		iwt97<<<blocksPerGrid, threadsPerBlock>>>(&output[i*col], &inputT[i*col], tempbank, col);

	cutilSafeCall(cudaFree(inputT));
}
