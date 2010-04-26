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
extern "C" void CUfwt97_2D_rgba(float* output, unsigned char* input, int row, int col);
extern "C" void CUiwt97_2D_rgba(unsigned char* output, float* input, int row, int col);

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
-1, 8, -1, \
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
0.125, 0.25, 0.125, \
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

void fwt2D_row(float* output, float* tempbank, int n, int dim, dim3 numBlocks, dim3 threadsPerBlock)
{
	predict1<<<numBlocks, threadsPerBlock>>>(output, n, dim);
	update1 <<<numBlocks, threadsPerBlock>>>(output, n, dim);
	predict2<<<numBlocks, threadsPerBlock>>>(output, n, dim);
	update2 <<<numBlocks, threadsPerBlock>>>(output, n, dim);
	scale   <<<numBlocks, threadsPerBlock>>>(output, n, dim);
	pack    <<<numBlocks, threadsPerBlock>>>(output, tempbank, n, dim);
	readOut <<<numBlocks, threadsPerBlock>>>(output, tempbank, n, dim);
}

void iwt2D_row(float* input, float* tempbank, int n, int dim, dim3 numBlocks, dim3 threadsPerBlock)
{
	UNpack <<<numBlocks, threadsPerBlock>>>(input, tempbank,n,dim);
	readOut <<<numBlocks, threadsPerBlock>>>(input, tempbank,n,dim);
	UNscale <<<numBlocks, threadsPerBlock>>>(input, n,dim);
	UNupdate2 <<<numBlocks, threadsPerBlock>>>(input, n,dim);
	UNpredict2<<<numBlocks, threadsPerBlock>>>(input, n,dim);
	UNupdate1 <<<numBlocks, threadsPerBlock>>>(input, n,dim);
	UNpredict1<<<numBlocks, threadsPerBlock>>>(input, n,dim);
}

#define TWO_D 1

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
	float* outputT;
	cutilSafeCall(cudaMalloc((void**)&tempbank, sizeof(float) * ((row>col)?row:col) * 4));
	cutilSafeCall(cudaMalloc((void**)&outputT, sizeof(float)*row*col));

	// execute the kernel
	for(i=0; i < row; i++){
#ifdef TWO_D
		readIn <<<numBlocks, threadsPerBlock>>>(&outputT[i*col], &input[i*col], col, dim);
		fwt2D_row(&outputT[i*col], tempbank, col, dim, numBlocks, threadsPerBlock);
#else
		readIn <<<numBlocks, threadsPerBlock>>>(&output[i*col], &input[i*col], col, dim);
		fwt2D_row(&output[i*col], tempbank, col, dim, numBlocks, threadsPerBlock);
#endif
	}

#ifdef TWO_D
	CUtranspose(output, outputT, col,row);

	for(i=0; i < col; i++){
		fwt2D_row(&output[i*row], tempbank, row, dim,numBlocks, threadsPerBlock);
	}
#endif

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
	float* inputT;
	cutilSafeCall(cudaMalloc((void**)&tempbank, sizeof(float) * ((row>col)?row:col) * 4));
	cutilSafeCall(cudaMalloc((void**)&inputT, sizeof(float)*row*col));

	// execute the kernel
#ifdef TWO_D
	for(i=0; i < col; i++){
		iwt2D_row(&input[i*row], tempbank,row,dim,numBlocks, threadsPerBlock);
	}

	CUtranspose(inputT, input, row,col);
#endif
	for(i=0; i < row; i++){
#ifdef TWO_D
		iwt2D_row(&inputT[i*col], tempbank,col,dim,numBlocks, threadsPerBlock);
		clamp <<<numBlocks, threadsPerBlock>>>(&output[i*col], &inputT[i*col], col,dim);
#else
		iwt2D_row(&input[i*col], tempbank,col,dim,numBlocks, threadsPerBlock);
		clamp <<<numBlocks, threadsPerBlock>>>(&output[i*col], &input[i*col], col,dim);
#endif
	}

	cutilSafeCall(cudaFree(inputT));
	cutilSafeCall(cudaFree(tempbank));
}


void CUfwt97_2D_rgba(float* output, unsigned char* input, int row, int col)
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
	float* outputT;
	cutilSafeCall(cudaMalloc((void**)&tempbank, sizeof(float) * ((row>col)?row:col)));
	cutilSafeCall(cudaMalloc((void**)&outputT, sizeof(float)*row*col*4));

	float* outputRT = &outputT[0];
	float* outputGT = &outputT[row*col];
	float* outputBT = &outputT[row*col*2];
	float* outputAT = &outputT[row*col*3];
	float* outputR = &output[0];
	float* outputG = &output[row*col];
	float* outputB = &output[row*col*2];
	float* outputA = &output[row*col*3];

	int threads = 512;
	int blocks = (row*col*4 + threads - 1) / threads;
	shuffle<<<blocks,threads>>>(outputT, input, col, row);
//	shuffle<<<blocks,threads>>>(output, input, col, row);

	// execute the kernel
	for(i=0; i < row; i++){
		fwt2D_row(&outputRT[i*col], tempbank, col, dim, numBlocks, threadsPerBlock);
		fwt2D_row(&outputGT[i*col], tempbank, col, dim, numBlocks, threadsPerBlock);
		fwt2D_row(&outputBT[i*col], tempbank, col, dim, numBlocks, threadsPerBlock);
		fwt2D_row(&outputAT[i*col], tempbank, col, dim, numBlocks, threadsPerBlock);
	}

	CUtranspose(outputR, outputRT, col,row);
	CUtranspose(outputG, outputGT, col,row);
	CUtranspose(outputB, outputBT, col,row);
	CUtranspose(outputA, outputAT, col,row);

	for(i=0; i < col; i++){
		fwt2D_row(&outputR[i*row], tempbank, row, dim,numBlocks, threadsPerBlock);
		fwt2D_row(&outputG[i*row], tempbank, row, dim,numBlocks, threadsPerBlock);
		fwt2D_row(&outputB[i*row], tempbank, row, dim,numBlocks, threadsPerBlock);
		fwt2D_row(&outputA[i*row], tempbank, row, dim,numBlocks, threadsPerBlock);
	}

	cutilSafeCall(cudaFree(tempbank));
	cutilSafeCall(cudaFree(outputT));
}

void CUiwt97_2D_rgba(unsigned char* output, float* input, int row, int col)
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
	float* inputT;
	cutilSafeCall(cudaMalloc((void**)&tempbank, sizeof(float) * ((row>col)?row:col)));
	cutilSafeCall(cudaMalloc((void**)&inputT, sizeof(float)*row*col*4));

	float* inputRT = &inputT[0];
	float* inputGT = &inputT[row*col];
	float* inputBT = &inputT[row*col*2];
	float* inputAT = &inputT[row*col*3];
	float* inputR = &input[0];
	float* inputG = &input[row*col];
	float* inputB = &input[row*col*2];
	float* inputA = &input[row*col*3];

	// execute the kernel
	for(i=0; i < col; i++){
		iwt2D_row(&inputR[i*row], tempbank,row,dim,numBlocks, threadsPerBlock);
		iwt2D_row(&inputG[i*row], tempbank,row,dim,numBlocks, threadsPerBlock);
		iwt2D_row(&inputB[i*row], tempbank,row,dim,numBlocks, threadsPerBlock);
		iwt2D_row(&inputA[i*row], tempbank,row,dim,numBlocks, threadsPerBlock);
	}

	CUtranspose(inputRT, inputR, row,col);
	CUtranspose(inputGT, inputG, row,col);
	CUtranspose(inputBT, inputB, row,col);
	CUtranspose(inputAT, inputA, row,col);

	for(i=0; i < row; i++){
		iwt2D_row(&inputRT[i*col], tempbank,col,dim,numBlocks, threadsPerBlock);
		iwt2D_row(&inputGT[i*col], tempbank,col,dim,numBlocks, threadsPerBlock);
		iwt2D_row(&inputBT[i*col], tempbank,col,dim,numBlocks, threadsPerBlock);
		iwt2D_row(&inputAT[i*col], tempbank,col,dim,numBlocks, threadsPerBlock);
	}

	int threads = 512;
	int blocks = (row*col*4 + threads - 1) / threads;
	UNshuffle<<<blocks,threads>>>(output, inputT, col, row);

	cutilSafeCall(cudaFree(inputT));
	cutilSafeCall(cudaFree(tempbank));
}


