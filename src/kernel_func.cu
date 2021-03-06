//http://www.embl.de/~gpau/index.html

#include <cutil_inline.h>
#include "TWODFWT.h"
#include "kernels.cu"

extern "C" void CUquantize(float* x, int Qlevel, int maxval, int len);
extern "C" void CUzeroOut(int* x, float threshold, int len);
extern "C" void CUtranspose(float* d_odata, float* d_idata, int col, int row);
extern "C" void CUsetToVal(unsigned char* x, int len, int val);
extern "C" void CUedgeDetect(unsigned char* input, unsigned char* output, int row, int col);
extern "C" void CUblur(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUbrighten(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUcontrast(unsigned char* output, unsigned char* input, int row, int col, float factor, float lum);
extern "C" void CUgreyscale(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUsaturate(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUfwt97_2D_rgba(int* outputInt, unsigned char* input, int row, int col);
extern "C" void CUiwt97_2D_rgba(unsigned char* output, int* input, int row, int col);
extern "C" void CUreduce(int size, mvec *d_idata, mvec *d_odata);
extern "C" mvec* CUmotVecFrame(unsigned char* prevImg, unsigned char* currImg, int height, int width);

void CUquantize(float* x, int Qlevel, int maxval, int len)
{
	int threadsPerBlock = 512;
	int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
	quantize<<<blocksPerGrid, threadsPerBlock>>>(x, Qlevel, maxval, len);
}

void CUzeroOut(int* x, float threshold, int len)
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

void CUcontrast(unsigned char* output, unsigned char* input, int row, int col, float factor, float lum)
{
	dim3 dimGrid(row/4+1, col/4+1);
	dim3 dimThreadBlock(16,16);
	contrast<<<dimGrid, dimThreadBlock>>>(input, output, row, col, factor, lum);
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

void fwt2D_row(float* output, float* tempbank, int n, int len, int dim, dim3 numBlocks, dim3 threadsPerBlock, int row, int col)
{
	predict<<<numBlocks, threadsPerBlock>>>(output, n, len, dim, -1.586134342f, col, row);
	update <<<numBlocks, threadsPerBlock>>>(output, n, len, dim, -0.05298011854f, col, row);
	predict<<<numBlocks, threadsPerBlock>>>(output, n, len, dim,  0.8829110762f, col, row);
	update <<<numBlocks, threadsPerBlock>>>(output, n, len, dim,  0.4435068522, col, row);
	scale  <<<numBlocks, threadsPerBlock>>>(output, n, len, dim,  0.869864452f, col, row);
	pack   <<<numBlocks, threadsPerBlock>>>(output, tempbank, n, len, dim);
	readOut<<<numBlocks, threadsPerBlock>>>(output, tempbank, n, len, dim);
}

void iwt2D_row(float* input, float* tempbank, int n, int len, int dim, dim3 numBlocks, dim3 threadsPerBlock, int row, int col)
{
	unpack <<<numBlocks, threadsPerBlock>>>(input, tempbank,n,len, dim);
	readOut<<<numBlocks, threadsPerBlock>>>(input, tempbank,n,len, dim);
	scale  <<<numBlocks, threadsPerBlock>>>(input, n, len, dim,  1.149604398f, col, row);
	update <<<numBlocks, threadsPerBlock>>>(input, n, len, dim, -0.4435068522f, col, row);
	predict<<<numBlocks, threadsPerBlock>>>(input, n, len, dim, -0.8829110762f, col, row);
	update <<<numBlocks, threadsPerBlock>>>(input, n, len, dim,  0.05298011854f, col, row);
	predict<<<numBlocks, threadsPerBlock>>>(input, n, len, dim,  1.586134342f, col, row);
}

void CUfwt97_2D_rgba(int* outputInt, unsigned char* input, int row, int col)
{
/*	if(row%2)
		row++;
	if(col%2)
		col++;
*/
	dim3 numBlocks(row/8+1,col/8+1,1);
	dim3 threadsPerBlock(8,8,4);
	int dim = row;

	float* tempbank;
	float* outputT;
	float* output;
	cutilSafeCall(cudaMalloc((void**)&tempbank, sizeof(float) * row*col*4));
	cutilSafeCall(cudaMalloc((void**)&outputT,  sizeof(float) * row*col*4));
	cutilSafeCall(cudaMalloc((void**)&output,   sizeof(float) * row*col*4));

	int threads = 512;
	int blocks = (row*col*4 + threads - 1) / threads;
	shuffle<<<blocks,threads>>>(outputT, input, col, row);


//	setToVal<<<blocks, threads>>>(output, row*col*4, 255);
//	setToVal<<<blocks, threads>>>(outputT, row*col*4, 100);
//	cudaError_t err;
//	err = cudaMemcpy(output, outputT, sizeof(float)*row*col*4, cudaMemcpyDeviceToDevice);
//	printf("%d\n",err);
//	fflush(stdout);
//	cutilSafeCall(cudaMemcpy(output, outputR, sizeof(float)*row*col*4, cudaMemcpyDeviceToDevice));

	// execute the kernel
	fwt2D_row(outputT, tempbank, row*col*4, col, dim, numBlocks, threadsPerBlock, col,row);

	CUtranspose(&output[0],         &outputT[0], col,row);
	CUtranspose(&output[row*col*1], &outputT[row*col*1], col,row);
	CUtranspose(&output[row*col*2], &outputT[row*col*2], col,row);
	CUtranspose(&output[row*col*3], &outputT[row*col*3], col,row);

	fwt2D_row(output, tempbank, row*col*4, row, dim, numBlocks, threadsPerBlock, col,row);
//#ifdef TWODFWT
	CUtranspose(&outputT[0],         &output[0], row,col);
	CUtranspose(&outputT[row*col*1], &output[row*col*1], row,col);
	CUtranspose(&outputT[row*col*2], &output[row*col*2], row,col);
	CUtranspose(&outputT[row*col*3], &output[row*col*3], row,col);

	fwt2D_row(outputT, tempbank, row*col*4, col, dim, numBlocks, threadsPerBlock, col,row);

	CUtranspose(&output[0],         &outputT[0], col,row);
	CUtranspose(&output[row*col*1], &outputT[row*col*1], col,row);
	CUtranspose(&output[row*col*2], &outputT[row*col*2], col,row);
	CUtranspose(&output[row*col*3], &outputT[row*col*3], col,row);

	fwt2D_row(output, tempbank, row*col*4, row, dim, numBlocks, threadsPerBlock, col,row);
//#endif
	roundArray<<<blocks,threads>>>(outputInt, output, col, row);

	cutilSafeCall(cudaFree(tempbank));
	cutilSafeCall(cudaFree(outputT));
	cutilSafeCall(cudaFree(output));
}

void CUiwt97_2D_rgba(unsigned char* output, int* inputInt, int row, int col)
{
/*	if(row%2)
		row++;
	if(col%2)
		col++;
*/
	dim3 numBlocks(row/8+1,col/8+1,1);
	dim3 threadsPerBlock(8,8,4);
	int dim = row;

	float* tempbank;
	float* input;
	float* inputT;
	cutilSafeCall(cudaMalloc((void**)&tempbank, sizeof(float) * row*col*4));
	cutilSafeCall(cudaMalloc((void**)&input, sizeof(float)*row*col*4));
	cutilSafeCall(cudaMalloc((void**)&inputT, sizeof(float)*row*col*4));

	int threads = 512;
	int blocks = (row*col*4 + threads - 1) / threads;
	intToFloat<<<blocks,threads>>>(input, inputInt, row,col);

	// execute the kernel
	iwt2D_row(input, tempbank, row*col*4,row,dim,numBlocks, threadsPerBlock, col,row);

	CUtranspose(&inputT[0]        , &input[0], row,col);
	CUtranspose(&inputT[row*col*1], &input[row*col*1], row,col);
	CUtranspose(&inputT[row*col*2], &input[row*col*2], row,col);
	CUtranspose(&inputT[row*col*3], &input[row*col*3], row,col);

	iwt2D_row(inputT, tempbank,row*col*4,col,dim,numBlocks, threadsPerBlock, col,row);
//#ifdef TWODFWT
	CUtranspose(&input[0]        , &inputT[0], col,row);
	CUtranspose(&input[row*col*1], &inputT[row*col*1], col,row);
	CUtranspose(&input[row*col*2], &inputT[row*col*2], col,row);
	CUtranspose(&input[row*col*3], &inputT[row*col*3], col,row);

	iwt2D_row(input, tempbank, row*col*4,row,dim,numBlocks, threadsPerBlock, col,row);

	CUtranspose(&inputT[0]        , &input[0], row,col);
	CUtranspose(&inputT[row*col*1], &input[row*col*1], row,col);
	CUtranspose(&inputT[row*col*2], &input[row*col*2], row,col);
	CUtranspose(&inputT[row*col*3], &input[row*col*3], row,col);

	iwt2D_row(inputT, tempbank,row*col*4,col,dim,numBlocks, threadsPerBlock, col,row);
//#endif
	UNshuffle<<<blocks,threads>>>(output, inputT, col, row);

	cutilSafeCall(cudaFree(input));
	cutilSafeCall(cudaFree(inputT));
	cutilSafeCall(cudaFree(tempbank));
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

void CUreduce(int size, mvec *d_idata, mvec *d_odata)
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

mvec* CUmotVecFrame(unsigned char* prevImg, unsigned char* currImg, int height, int width)
{
	int blockDimX = CEIL(width/8.0f);
	int blockDimY = CEIL(height/8.0f);
	mvec* vecs = (mvec*)malloc(sizeof(mvec) * blockDimX * blockDimY );

	int ucMemsize = sizeof(unsigned char)*height*width*4;
	int siMemsize = sizeof(unsigned char)*height*width*4;

	//Allocate space and move everything onto the GPU
	unsigned char* CUcurrImg;
	cutilSafeCall( cudaMalloc((void**)&CUcurrImg, ucMemsize) );
	cutilSafeCall( cudaMemcpy(CUcurrImg, currImg, ucMemsize, cudaMemcpyHostToDevice) );
	unsigned char* CUprevImg;
	cutilSafeCall( cudaMalloc((void**)&CUprevImg, siMemsize) );
	cutilSafeCall( cudaMemcpy(CUprevImg, prevImg, siMemsize, cudaMemcpyHostToDevice) );

	//Allocate space for all the potential mvecs and min mvecs
	mvec* CUminvecs;
	cutilSafeCall(cudaMalloc((void**)&CUminvecs , sizeof(mvec)*blockDimX*blockDimY));
	mvec* CUallmvecs;
	cutilSafeCall(cudaMalloc((void**)&CUallmvecs, sizeof(mvec)*blockDimX*blockDimY*17*17));

	// calculate all blockDimX*blockDimY*17*17 mvecs
	dim3 blockSize(blockDimX, blockDimY); // each motion vector gets its own block
	dim3 threadSize(17,17);				  // each block has enough thread for each vector (8+1+8)^2

	findAllVals<<<blockSize, threadSize>>>(CUallmvecs, blockDimY, blockDimX, CUprevImg, CUcurrImg, height, width);
/*
	FILE* debugOut = fopen("debug.txt","w");

	mvec* debug = (mvec*)malloc(sizeof(mvec)*blockDimX*blockDimY*17*17);
	cutilSafeCall(cudaMemcpy(debug, CUallmvecs, sizeof(mvec)*blockDimX*blockDimY*17*17, cudaMemcpyDeviceToHost));
	for(int j=0; j < blockDimX*blockDimY; j++){
		for(int i=0; i < 17*17; i++){
			fprintf(debugOut, "%d\t%d\t%d\n",debug[i + j * 17*17].x,debug[i + j * 17*17].y,debug[i + j * 17*17].diff);
		}
	}
	free(debug);

	fclose(debugOut);
*/
	// reduce each block
	for(int i=0; i < blockDimX*blockDimY; i++)
		CUreduce(17*17, &CUallmvecs[17*17*i], &CUminvecs[i]);

	// move reduced values to CPU
	cutilSafeCall(cudaMemcpy(vecs, CUminvecs, sizeof(mvec)*blockDimX * blockDimY, cudaMemcpyDeviceToHost) );

	cutilSafeCall(cudaFree(CUallmvecs));
	cutilSafeCall(cudaFree(CUminvecs));
	cutilSafeCall(cudaFree(CUprevImg));
	cutilSafeCall(cudaFree(CUcurrImg));
	return vecs;
}
