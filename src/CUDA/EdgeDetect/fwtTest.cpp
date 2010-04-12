#include <cutil_inline.h>
#include <cuda.h>
#include <cstdlib>
#include <cstdio>
#include "utility.c"

extern "C" void edgeDetectGPU(unsigned char* input, unsigned char* output, int row, int col);
extern "C" void CUfwt97(float* output, unsigned char* input, float* tempbank, int n);
extern "C" void CUfwt97_2D(float* output, unsigned char* input, float* tempbank, int row, int col);
extern "C" void CUiwt97(unsigned char* output, float* input, float* tempbank, int n);
extern "C" void CUiwt97_2D(unsigned char* output, float* input, float* tempbank, int row, int col);
extern "C" void CUquantize(float* x, int Qlevel, int maxval, int len);
extern "C" void CUsetToVal(unsigned char* x, int len, int val);

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("Usage: %s image_file out_file Qlevel\n", argv[0]);
		exit(0); 
	}
	int row, col, color;
	unsigned char** pic = alloc_read_image(argv[1],&row,&col,&color);
	unsigned char ** picOUT = allocate_uchar (col,row);

	int len = row*col;
/*	if( (int)(log(len)/log(2)) != (log(len)/log(2))){
		printf("image length must be a power of 2\n%d\t%2.30f\n",(int)(log(len)/log(2)), (log(len)/log(2)));
		exit(0);
	}
*/

	float* CUpicF;
	float* CUtempbank;
	unsigned char* CUpic;
	unsigned char* CUpicOUT;

	cutilSafeCall(cudaMalloc((void**)&CUpicF, sizeof(float)*len));
	cutilSafeCall(cudaMalloc((void**)&CUtempbank, sizeof(float)*len));
	cutilSafeCall(cudaMalloc((void**)&CUpic, sizeof(unsigned char)*len));
	cutilSafeCall(cudaMalloc((void**)&CUpicOUT, sizeof(unsigned char)*len));

	cutilSafeCall(cudaMemcpy(CUpic, pic[0], sizeof(unsigned char)*len, cudaMemcpyHostToDevice));

	CUfwt97_2D(CUpicF, CUpic, CUtempbank, row,col);

	int Qlevel = atoi(argv[3]);

	CUquantize(CUpicF, Qlevel, 511, len);

	float* transformed = (float*)malloc(sizeof(float)*len);
	cutilSafeCall(cudaMemcpy(transformed, CUpicF, sizeof(float)*len, cudaMemcpyDeviceToHost));
	int i, coeff=0, zeroCoeff=0;
	for(i=0; i < row*col; i++){
		if(transformed[i]==0)
			zeroCoeff++;
		coeff++;
	}
	printf("pct nonzero coeff: %f\n", 100*(coeff-zeroCoeff)/(float)coeff);
	free(transformed);

//	CUsetToVal(CUpicOUT, len, 0);

	CUiwt97_2D(CUpicOUT, CUpicF, CUtempbank, row,col);

	cutilSafeCall(cudaMemcpy(picOUT[0], CUpicOUT, sizeof(unsigned char)*len, cudaMemcpyDeviceToHost));

	save_image(picOUT,argv[2],row,col,color);

	free2Duchar(pic);
	free2Duchar(picOUT);
	cutilSafeCall(cudaFree(CUpicF));
	cutilSafeCall(cudaFree(CUtempbank));
	cutilSafeCall(cudaFree(CUpic));
	cutilSafeCall(cudaFree(CUpicOUT));
}
