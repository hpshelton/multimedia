#include <sys/time.h>
#include <cutil_inline.h>
#include <cfloat>
#include <vector>
#include <cstring>
#include "utility.c"

#define FRAMESIZE 38016	//framesize in bytes for glasgow100.qcif
#define ROWS 144
#define COLS 176

#define index(a,b) [(a) + ROWS * (b)]
#define ABS(a) ((a)<0 ? -(a) : (a))

/*
__global__ void subFrames(unsigned char* frameDiff, const unsigned char* currFrame, const unsigned char* prevFrame, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
		frameDiff[i] = (unsigned char)((currFrame[i] - prevFrame[i] + 255)/2);
	}
}
*/

__global__ void getMotVec(unsigned char* currFrame, unsigned char* prevFrame, int* vector) {
	int Yorigin = (blockIdx.y * blockDim.y + threadIdx.y)*16;
	int Xorigin = (blockIdx.x * blockDim.x + threadIdx.x)*16;

	int GPUy = (blockIdx.y * blockDim.y + threadIdx.y);
	int GPUx = (blockIdx.x * blockDim.x + threadIdx.x);

	int i, j;
	float diff, minDiff=FLT_MAX;

	if(GPUx < COLS/16 && GPUy < ROWS/16){
	/*
		for(i=0; i < 99*2; i++)
			vector[i]=i;
	*/
		vector[(GPUy + (ROWS/16)*GPUx)*2] = 7;
		vector[(GPUy + (ROWS/16)*GPUx)*2+1] = 7;

		for(int xOffset=-15; xOffset < 16; xOffset++) {
			for(int yOffset=-15; yOffset < 16; yOffset++) {
				diff=0;
				for(i=0; i < 16; i++) {
					for(j=0; j < 16; j++) {
						if ( (Xorigin + i + xOffset) < 0 || (COLS-1) < (Xorigin + i + xOffset) ){
							diff += currFrame index(Yorigin + j, Xorigin + i);
						}
						else if ( (Yorigin + j + yOffset) < 0 || (ROWS-1) < (Yorigin + j + yOffset) ){
							diff += currFrame index(Yorigin + j, Xorigin + i);
						}
						else{
							diff += abs(currFrame index(Yorigin + j, Xorigin + i) - prevFrame index(Yorigin + j + yOffset, Xorigin + i + xOffset));
						}
					}
				}
	/*			if(diff<minDiff){
					minDiff=diff;
					vector[(Xorigin/16 + COLS*Yorigin/16)*2] = xOffset;
					vector[(Xorigin/16 + COLS*Yorigin/16)*2+1] = yOffset;
				}
				else if(diff==minDiff){
					if(sqrt(xOffset*xOffset + yOffset*yOffset) < sqrt(vector[(Xorigin/16 + COLS*Yorigin/16)*2]*vector[(Xorigin/16 + COLS*Yorigin/16)*2] + vector[(Xorigin/16 + COLS*Yorigin/16)*2+1]*vector[(Xorigin/16 + COLS*Yorigin/16)*2+1])){
						minDiff=diff;
						vector[(Xorigin/16 + COLS*Yorigin/16)*2] = xOffset;
						vector[(Xorigin/16 + COLS*Yorigin/16)*2+1] = yOffset;
					}
				}
	*/		}
		}
	}
}

int* getMotionVector(unsigned char** currFrame, unsigned char** prevFrame, int Xorigin, int Yorigin) {
	int xOffset, yOffset, i, j;
	float diff;
	float minDiff = FLT_MAX;
	static int vector[2];

	for(xOffset=-15; xOffset < 16; xOffset++) {
		for(yOffset=-15; yOffset < 16; yOffset++) {
			diff=0;
			for(i=0; i < 16; i++) {
				for(j=0; j < 16; j++) {
					if ( (Xorigin + i + xOffset) < 0 || (COLS-1) < (Xorigin + i + xOffset) ){
						diff += abs(currFrame[Yorigin + j][Xorigin + i]);
					}
					else if ( (Yorigin + j + yOffset) < 0 || (ROWS-1) < (Yorigin + j + yOffset) ){
						diff += abs(currFrame[Yorigin + j][Xorigin + i]);
					}
					else{
						diff += abs(currFrame[Yorigin + j][Xorigin + i] - prevFrame[Yorigin + j + yOffset][Xorigin + i + xOffset]);
					}
				}
			}
			if(diff<minDiff){
				minDiff=diff;
				vector[0] = xOffset;
				vector[1] = yOffset;
			}
			else if(diff==minDiff){
				if(sqrt(xOffset*xOffset + yOffset*yOffset) < sqrt(vector[0]*vector[0] + vector[1]*vector[1])){
					minDiff=diff;
					vector[0] = xOffset;
					vector[1] = yOffset;
				}
			}
		}
	}
	return vector;
}

int main(int argc, char* argv[])
{
	if(argc != 4){
		printf("usage: frameDiff infile outfileCPU outfileGPU\n");
		return -1;
	}
	timeval t1, t2;

	std::vector<int> vecX;
	std::vector<int> vecY;
	std::vector<int> vecU;
	std::vector<int> vecV;

	FILE* infile =  fopen(argv[1],"r");
	FILE* outfileCPU = fopen(argv[2],"w");
	FILE* outfileGPU = fopen(argv[3],"w");
	FILE* vecOut;
	int i,j, k;
	
	fseek(infile, 0, SEEK_END);
	int numFrames = ftell(infile)/FRAMESIZE;
	rewind(infile);
	
	unsigned char** prevFrameCPU = allocate_uchar (COLS,ROWS);
	unsigned char** currFrameCPU = allocate_uchar (COLS,ROWS);
	unsigned char** frameDiffCPU = allocate_uchar (COLS,ROWS);

	for(i=0; i < ROWS; i++){
		for(j=0; j < COLS; j++){
			prevFrameCPU[i][j]=0;
		}
	}

	int* vector;
	int frameNum=0;
	// total hack, but char* filename="mvec000.txt"; would segfault when changing numbers
	char* filename=(char*)malloc(sizeof(char)*12);
	filename[0]='m';
	filename[1]='v';
	filename[2]='e';
	filename[3]='c';
	filename[4]='0';
	filename[5]='0';
	filename[6]='0';
	filename[7]='.';
	filename[8]='t';
	filename[9]='x';
	filename[10]='t';
	filename[11]='\0';

    gettimeofday(&t1, NULL);
	while(fread(currFrameCPU[0], sizeof(unsigned char), ROWS*COLS, infile)){

/*		for(i=0; i < COLS/16; i++){
			for(j=0; j < ROWS/16; j++){
				vector = getMotionVector(currFrameCPU, prevFrameCPU, i*16, j*16);
				vecX.push_back(i*16+8);
				vecY.push_back(j*16+8);
				vecU.push_back(vector[0]);
				vecV.push_back(vector[1]);
//				fprintf(outfileCPU, "%d\t%d\t%d\t%d\t%d\n",frameNum, i, j, vector[0], vector[1]); 
			}
		}

		filename[4] = '0'+(frameNum/100)%10;
		filename[5] = '0'+(frameNum/10)%10;
		filename[6] = '0'+(frameNum)%10;

		vecOut = fopen(filename, "w");
		fprintf(vecOut, "X=[");
		for(k=0; k < vecX.size(); k++){
			fprintf(vecOut, "%d ",vecX[k]);
		}
		fprintf(vecOut, "];\nY=[");
		for(k=0; k < vecY.size(); k++){
			fprintf(vecOut, "%d ",vecY[k]);
		}
		fprintf(vecOut, "];\nU=[");
		for(k=0; k < vecU.size(); k++){
			fprintf(vecOut, "%d ",vecU[k]);
		}
		fprintf(vecOut, "];\nV=[");
		for(k=0; k < vecV.size(); k++){
			fprintf(vecOut, "%d ",vecV[k]);
		}
		fprintf(vecOut,"];");
		fclose(vecOut);

		vecX.clear();
		vecY.clear();
		vecU.clear();
		vecV.clear();
*/
		frameNum++;
		free2Duchar(prevFrameCPU);
		prevFrameCPU = currFrameCPU;
		currFrameCPU = allocate_uchar (COLS,ROWS);
		fseek(infile, FRAMESIZE*(frameNum), SEEK_SET);
		
		printf("\r%3.0f%%",100*frameNum/(float)numFrames);
		fflush(stdout);
	}
    gettimeofday(&t2, NULL);

	float elapsedTime;
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	printf("\nCPU Time Elapsed: %fms\n",elapsedTime);

	fclose(outfileCPU);
	rewind(infile);
	
	int numBlocks = ROWS/16 * COLS/16;
	unsigned char* prevFrameGPU;
	unsigned char* currFrameGPU;
	int* vectorGPU;
	cutilSafeCall(cudaMalloc((void**) &prevFrameGPU, sizeof(unsigned char)*ROWS*COLS));
	cutilSafeCall(cudaMalloc((void**) &currFrameGPU, sizeof(unsigned char)*ROWS*COLS));
	cutilSafeCall(cudaMalloc((void**) &vectorGPU, sizeof(int)*2*numBlocks));
	vector = (int*)malloc(sizeof(int)*2*numBlocks);

	for(i=0; i < ROWS; i++){
		for(j=0; j < COLS; j++){
			prevFrameCPU[i][j]=0;
		}
	}
	cutilSafeCall(cudaMemcpy(prevFrameGPU, prevFrameCPU[0], sizeof(unsigned char)*COLS*ROWS, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(currFrameGPU, prevFrameCPU[0], sizeof(unsigned char)*ROWS*COLS, cudaMemcpyHostToDevice));
	// warm up the kernel
	getMotVec<<<128,128>>>(currFrameGPU, prevFrameGPU, vectorGPU);

	frameNum=0;

    gettimeofday(&t1, NULL);
	while(fread(currFrameCPU[0], sizeof(unsigned char), ROWS*COLS, infile)){
			
		cutilSafeCall(cudaMemcpy(currFrameGPU, currFrameCPU[0], sizeof(unsigned char)*ROWS*COLS, cudaMemcpyHostToDevice));

		getMotVec<<<128,128>>>(currFrameGPU, prevFrameGPU, vectorGPU);
		cutilSafeCall(cudaMemcpy(vector, vectorGPU, sizeof(int)*2*numBlocks, cudaMemcpyDeviceToHost));
		for(i=0; i < COLS/16; i++){
			for(j=0; j < ROWS/16; j++){
				fprintf(outfileGPU, "%d\t%d\t%d\t%d\t%d\n",frameNum, i, j, vector[(j+i*8)*2], vector[(j+i*8)*2+1]); 
			}
		}

		cutilSafeCall(cudaFree(prevFrameGPU));
		prevFrameGPU = currFrameGPU;
		cutilSafeCall(cudaMalloc((void**) &currFrameGPU, sizeof(unsigned char)*ROWS*COLS));

		frameNum++;
		fseek(infile, FRAMESIZE*(frameNum), SEEK_SET);
		
		printf("\r%3.0f%%",100*frameNum/(float)numFrames);
		fflush(stdout);

	}
    gettimeofday(&t2, NULL);
    
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	printf("\nGPU Time Elapsed: %fms\n",elapsedTime);

	fclose(infile);
	fclose(outfileGPU);
	
	cutilSafeCall(cudaFree(prevFrameGPU));
	cutilSafeCall(cudaFree(currFrameGPU));
	cutilSafeCall(cudaFree(vectorGPU));

//	free2Duchar(prevFrameCPU);
//	free2Duchar(currFrameCPU);
	free2Duchar(frameDiffCPU);
	free(vector);
	
	return 0;
}
