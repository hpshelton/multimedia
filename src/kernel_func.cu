#include <cutil_inline.h>
#include "kernels.cu"

extern "C" void CUquantize(float* x, int Qlevel, int maxval, int len);
extern "C" void CUzeroOut(float* x, float threshold, int len);
extern "C" void CUtranspose(float* d_odata, float* d_idata, int col, int row);
extern "C" void CUsetToVal(unsigned char* x, int len, int val);
extern "C" void CUedgeDetect(unsigned char* input, unsigned char* output, int row, int col);
extern "C" void CUblur(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUbrighten(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUgreyscale(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUsaturate(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUfwt97_2D(float* output, unsigned char* input, int row, int col);
extern "C" void CUiwt97_2D(unsigned char* output, float* input, int row, int col);

void CUquantize(float* x, int Qlevel, int maxval, int len)
{
	int threadsPerBlock = 512;
	int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
	quantize<<<blocksPerGrid, threadsPerBlock>>>(x, Qlevel, maxval, len);
}

void CUzeroOut(float* x, float threshold, int len)
{
	int threadsPerBlock = 512;
	int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
	zeroOut<<<blocksPerGrid, threadsPerBlock>>>(x, threshold, len);
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

void CUfwt97_2D(float* output, unsigned char* input, int row, int col)
{
	if(row%2)
		row++;
	if(col%2)
		col++;

	int i;

	int blocknum = sqrt( (row>col)?row:col / 256 + 1)+1;
	dim3 numBlocks(blocknum,blocknum);
	dim3 threadsPerBlock(16,16);
	int dim = 16*blocknum;

	float* tempbank;
	cutilSafeCall(cudaMalloc((void**)&tempbank, sizeof(float) * ((row>col)?row:col) * 4));

	float* outputT;
	cutilSafeCall(cudaMalloc((void**)&outputT, sizeof(float)*row*col));

	// execute the kernel
	for(i=0; i < row; i++){
		readIn  <<<numBlocks, threadsPerBlock>>>(&outputT[i*col], &input[i*col], col, dim);
		predict1<<<numBlocks, threadsPerBlock>>>(&outputT[i*col], col, dim);
		update1 <<<numBlocks, threadsPerBlock>>>(&outputT[i*col], col, dim);
		predict2<<<numBlocks, threadsPerBlock>>>(&outputT[i*col], col, dim);
		update2 <<<numBlocks, threadsPerBlock>>>(&outputT[i*col], col, dim);
		scale   <<<numBlocks, threadsPerBlock>>>(&outputT[i*col], col, dim);
		pack    <<<numBlocks, threadsPerBlock>>>(&outputT[i*col], tempbank, col, dim);
		readOut <<<numBlocks, threadsPerBlock>>>(&outputT[i*col], tempbank, col, dim);
	}
		
	CUtranspose(output, outputT, col,row);

	for(i=0; i < col; i++){
		predict1<<<numBlocks, threadsPerBlock>>>(&output[i*row], row, dim);
		update1 <<<numBlocks, threadsPerBlock>>>(&output[i*row], row, dim);
		predict2<<<numBlocks, threadsPerBlock>>>(&output[i*row], row, dim);
		update2 <<<numBlocks, threadsPerBlock>>>(&output[i*row], row, dim);
		scale   <<<numBlocks, threadsPerBlock>>>(&output[i*row], row, dim);
		pack    <<<numBlocks, threadsPerBlock>>>(&output[i*row], tempbank, row, dim);
		readOut <<<numBlocks, threadsPerBlock>>>(&output[i*row], tempbank, row, dim);
	}

	cutilSafeCall(cudaFree(tempbank));
	cutilSafeCall(cudaFree(outputT));
}

void CUiwt97_2D(unsigned char* output, float* input, int row, int col)
{
	if(row%2)
		row++;
	if(col%2)
		col++;

	int i;

	int blocknum = sqrt( (row>col)?row:col / 256 + 1)+1;
	dim3 numBlocks(blocknum,blocknum);
	dim3 threadsPerBlock(16,16);
	int dim = 16*blocknum;

	float* tempbank;
	cutilSafeCall(cudaMalloc((void**)&tempbank, sizeof(float) * ((row>col)?row:col) * 4));

	float* inputT;
	cutilSafeCall(cudaMalloc((void**)&inputT, sizeof(float)*row*col));

	// execute the kernel

	for(i=0; i < col; i++){
		UNpack    <<<numBlocks, threadsPerBlock>>>(&input[i*row], tempbank,row,dim);
		readOut   <<<numBlocks, threadsPerBlock>>>(&input[i*row], tempbank,row,dim);
		UNscale   <<<numBlocks, threadsPerBlock>>>(&input[i*row], row,dim);
		UNupdate2 <<<numBlocks, threadsPerBlock>>>(&input[i*row], row,dim);
		UNpredict2<<<numBlocks, threadsPerBlock>>>(&input[i*row], row,dim);
		UNupdate1 <<<numBlocks, threadsPerBlock>>>(&input[i*row], row,dim);
		UNpredict1<<<numBlocks, threadsPerBlock>>>(&input[i*row], row,dim);
	}

	CUtranspose(inputT, input, row,col);

	for(i=0; i < row; i++){
		UNpack    <<<numBlocks, threadsPerBlock>>>(&inputT[i*col], tempbank,col,dim);
		readOut   <<<numBlocks, threadsPerBlock>>>(&inputT[i*col], tempbank,col,dim);
		UNscale   <<<numBlocks, threadsPerBlock>>>(&inputT[i*col], col,dim);
		UNupdate2 <<<numBlocks, threadsPerBlock>>>(&inputT[i*col], col,dim);
		UNpredict2<<<numBlocks, threadsPerBlock>>>(&inputT[i*col], col,dim);
		UNupdate1 <<<numBlocks, threadsPerBlock>>>(&inputT[i*col], col,dim);
		UNpredict1<<<numBlocks, threadsPerBlock>>>(&inputT[i*col], col,dim);
		clamp     <<<numBlocks, threadsPerBlock>>>(&output[i*col], &inputT[i*col], col,dim);
	}

	cutilSafeCall(cudaFree(inputT));
	cutilSafeCall(cudaFree(tempbank));
}
