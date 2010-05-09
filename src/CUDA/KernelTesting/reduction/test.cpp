#include "mvec.h"
#include <cutil_inline.h>
#include <cuda.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>

extern "C" void reduce(int size, mvec *d_idata, mvec *d_odata);

void printMvec(mvec A){
	printf("%6d\t%6d\t%6d\n", A.x, A.y, A.diff);
}

int main(int argc, char* argv[])
{
	if(argc != 2){
		printf("try again\n");
		return 0;
	}
	
	int ARRSIZE = atoi(argv[1]);

	mvec* mvecs = (mvec*)malloc(sizeof(mvec)*ARRSIZE);
	mvec* mvecsOut = (mvec*)malloc(sizeof(mvec));

	srand ( time(NULL) );


	for(int i=0; i < ARRSIZE; i++){
		mvecs[i].x = i;
		mvecs[i].y = i;
		mvecs[i].diff = rand() % 1024 ;
	}
	mvecs[ARRSIZE-1].diff=1;

	mvec* CUvecs;
	mvec* CUvecsOut;
	cutilSafeCall(cudaMalloc((void**)&CUvecs , sizeof(mvec)*ARRSIZE));
	cutilSafeCall(cudaMalloc((void**)&CUvecsOut , sizeof(mvec)));

	cutilSafeCall(cudaMemcpy(CUvecs, mvecs, sizeof(mvec)*ARRSIZE, cudaMemcpyHostToDevice));
	reduce(ARRSIZE, CUvecs, CUvecsOut);
	cutilSafeCall(cudaMemcpy(mvecsOut, CUvecsOut, sizeof(mvec), cudaMemcpyDeviceToHost));

	printf("originals:\n");
	for(int i=0; i < ARRSIZE; i++){
		printMvec(mvecs[i]);
	}
	printf("\nmin:\n");
	printMvec(mvecsOut[0]);

	cutilSafeCall(cudaFree(CUvecs));
	cutilSafeCall(cudaFree(CUvecsOut));
	free(mvecs);
	free(mvecsOut);
	return 0;
}
