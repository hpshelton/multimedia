#include <sys/time.h>
#if UNIX
    #include "/home/adam/NVIDIA_GPU_Computing_SDK/C/common/inc/cutil_inline.h"
#else
    #include "/Developer/GPU Computing/C/common/inc/cutil_inline.h"
#endif 
#include <cuda.h>
#include <cfloat>
#include <vector>
#include <cstring>
#include "utility.c"

#define index(a,b) [(a) + ROWS * (b)]
#define ABS(a) ((a)<0 ? -(a) : (a))
#define CLAMP(a) ((a>255) ? 255 : ((a<0) ? 0 : a))

extern "C" void edgeDetectGPU(unsigned char* input, unsigned char* output, int row, int col);

void edgeDetectCPU(unsigned char** input, unsigned char** output, int width, int height, int row, int col)
{
	float coeff[3][3] = {{-1, -1, -1},
                         {-1,  8, -1},
                         {-1, -1, -1}};

	int i, j;
	float convSum=0;
	for(i=-1; i < 2; i++){
		for(j=-1; j < 2; j++){
			if( 0<=(height+i) && (height+i) < row && 0 <= (width+j) && (width+j) < col ){
				convSum += coeff[i+1][j+1]*input[height+i][width+j];
			}
		}
	}
	output[height][width] = CLAMP(convSum);
}

int main(int argc, char* argv[])
{
	if(argc != 4){
		printf("usage: EdgeDetect infile outfileCPU outfileGPU\n");
		return -1;
	}

////////////////////////
//  CPU Calculations  //
////////////////////////

	int i,j;
	int row, col, color;

	unsigned char** input;
	unsigned char** output;

	input = alloc_read_image(argv[1], &row, &col, &color);
	output = allocate_uchar(col, row);

	for(i=0; i < row; i++){
		for(j=0; j < col; j++){
			edgeDetectCPU(input, output, j, i, row, col);
		}
	}

	save_image(output, argv[2], row, col, color);

////////////////////////
//  GPU Calculations  //
////////////////////////

	unsigned char* inputGPU;
	unsigned char* outputGPU;
	unsigned char** GPUresult;

	cutilSafeCall(cudaMalloc((void**)&inputGPU, sizeof(unsigned char)*row*col));
	cutilSafeCall(cudaMalloc((void**)&outputGPU, sizeof(unsigned char)*row*col));
	GPUresult = allocate_uchar(col, row);

	cutilSafeCall(cudaMemcpy(inputGPU, input[0], sizeof(unsigned char)*col*row, cudaMemcpyHostToDevice));
	edgeDetectGPU(inputGPU, outputGPU, row, col);
	cutilSafeCall(cudaMemcpy(GPUresult[0], outputGPU, sizeof(unsigned char)*col*row, cudaMemcpyDeviceToHost));

	save_image(GPUresult, argv[3], row, col, color);

////////////////////////
//   Memory Cleanup   //
////////////////////////

	cutilSafeCall(cudaFree(inputGPU));
	cutilSafeCall(cudaFree(outputGPU));
	free2Duchar(input);
	free2Duchar(output);
	free2Duchar(GPUresult);
	
	return 0;
}
