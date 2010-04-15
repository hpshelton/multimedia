#include <cutil_inline.h>
#include "kernels.cu"

extern "C" void CUquantize(float* x, int Qlevel, int maxval, int len);
extern "C" void CUtranspose(float* d_odata, float* d_idata, int col, int row);
extern "C" void CUsetToVal(unsigned char* x, int len, int val);
extern "C" void CUedgeDetect(unsigned char* input, unsigned char* output, int row, int col);
extern "C" void CUblur(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUbrighten(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUgreyscale(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUsaturate(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUfwt97   (float* output, unsigned char* input, float* tempbank, int n);
extern "C" void CUiwt97   (unsigned char* output, float* input, float* tempbank, int n);
extern "C" void CUiwt97_2D(unsigned char* output, float* input, float* tempbank, int row, int col);
extern "C" void CUfwt97_2D(float* output, unsigned char* input, float* tempbank, int row, int col);

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

void CUedgeDetect(unsigned char* output, unsigned char* input, int row, int col)
{
	dim3 dimGrid(row/4+1, col/4+1);
	dim3 dimThreadBlock(16,16);

	float coeff[9]= {-1, -1, -1, \
					 -1,  8, -1, \
					 -1, -1, -1};
	float* CUcoeff;
	cutilSafeCall(cudaMalloc((void**)&CUcoeff, sizeof(float)*9));
	cutilSafeCall(cudaMemcpy(CUcoeff, coeff, sizeof(float)*9, cudaMemcpyHostToDevice));

	conv3x3<<<dimGrid, dimThreadBlock>>>(input, output, row, col, CUcoeff);
	cutilSafeCall(cudaFree(CUcoeff));
}

void CUblur(unsigned char* output, unsigned char* input, int row, int col)
{
	dim3 dimGrid(row/4+1, col/4+1);
	dim3 dimThreadBlock(16,16);

	float coeff[9]= { 0.0625, 0.125, 0.0625, \
					  0.125,  0.25,  0.125,  \
					  0.0625, 0.125, 0.0625 };

	float* CUcoeff;
	cutilSafeCall(cudaMalloc((void**)&CUcoeff, sizeof(float)*9));
	cutilSafeCall(cudaMemcpy(CUcoeff, coeff, sizeof(float)*9, cudaMemcpyHostToDevice));

	conv3x3<<<dimGrid, dimThreadBlock>>>(input, output, row, col, CUcoeff);
	cutilSafeCall(cudaFree(CUcoeff));
}

void CUbrighten(unsigned char* output, unsigned char* input, int row, int col, float factor)
{
	dim3 dimGrid(row/4+1, col/4+1);
	dim3 dimThreadBlock(16,16);

	brighten<<<dimGrid, dimThreadBlock>>>(input, output, row, col, factor);
}

void CUgreyscale(unsigned char* output, unsigned char* input, int row, int col)
{
	dim3 dimGrid(row/16+1, col/16+1);
	dim3 dimThreadBlock(16,16);

	greyscale<<<dimGrid, dimThreadBlock>>>(input, output, row, col);
}

void CUsaturate(unsigned char* output, unsigned char* input, int row, int col, float factor)
{
	dim3 dimGrid(row/16+1, col/16+1);
	dim3 dimThreadBlock(16,16);

	saturate<<<dimGrid, dimThreadBlock>>>(input, output, row, col, factor);
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

	CUtranspose(output, outputT, col,row);

	threadsPerBlock = row;
	blocksPerGrid = (row + threadsPerBlock - 1) / threadsPerBlock;

	for(i=0; i < col; i++)
		fwt97<<<blocksPerGrid, threadsPerBlock>>>(&output[i*row], tempbank, row);

	cutilSafeCall(cudaFree(outputT));
}
void CUiwt97_2D(unsigned char* output, float* input, float* tempbank, int row, int col)
{
	int i;
	int threadsPerBlock = row;
	int blocksPerGrid = (row + threadsPerBlock - 1) / threadsPerBlock;
	float* inputT;
	cutilSafeCall(cudaMalloc((void**)&inputT, sizeof(float)*row*col));

	// execute the kernel

	for(i=0; i < col; i++)
		iwt97<<<blocksPerGrid, threadsPerBlock>>>(&input[i*row], tempbank, row);
	CUtranspose(inputT, input, row,col);

	threadsPerBlock = col;
	blocksPerGrid = (col + threadsPerBlock - 1) / threadsPerBlock;

	for(i=0; i < row; i++)
		iwt97<<<blocksPerGrid, threadsPerBlock>>>(&output[i*col], &inputT[i*col], tempbank, col);

	cutilSafeCall(cudaFree(inputT));
}
