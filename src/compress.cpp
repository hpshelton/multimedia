#include "mainwindow.h"
#include "dwt97.h"
#include "utility.h"
#include "TWODFWT.h"

#define NUM_SYMBOLS 512 // -256 -> 255

void RoundArray(int* out, float* in, int len)
{
	int i;
	for(i=0; i < len; i++){
		out[i] = (int)(in[i]+0.5);
	}
}

void intToFloat(float* out, int* in, int len)
{
	int i;
	for(i=0; i < len; i++){
		out[i] = in[i];
	}
}

int* Encoder::compress_image(QImage* img, float compression, bool CUDA, unsigned long* numBytes)
{
	int width = img->width();
	int height = img->height();
	*numBytes = height*width*4;
	int* compressed = (int*)malloc(sizeof(int) * (*numBytes));
	float threshold;

#ifdef TWODFWT
	if(compression < 50)
		threshold = 0;
	else if(compression < 77.703568)
		threshold = 0.000116*pow(compression,3) - 0.019118*pow(compression,2) + 1.108995*compression - 21.184998;
	else if(compression < 91.641708)
		/*PROBLEM*/
		threshold = 0.000046386058562*pow(compression,6)-0.023471363133066*pow(compression,5)+4.94695855894703*pow(compression,4)-555.891027747718*pow(compression,3)+35124.430525745*pow(compression,2)-1183223.56408731*compression+16601453.5673362;
	else if(compression < 92.569443)
		threshold = 0.873366*compression + 12.532677;
	else if(compression < 94.001099)
		threshold = 153.426432*pow(compression,4) - 57142.000968*pow(compression,3) + 7980724.953524*pow(compression,2) - 495389786.739970*compression + 11531433455.627500;
	else
		threshold = 0.0000000000320307431396312 * pow(E,0.315611691031623*compression);
#else
	if(compression < 19.4963201)
		threshold = 0;
	else if(compression <= 52.7)
		threshold = compression*0.045110 - 0.879479;
	else if(compression < 68.29137)
		threshold = 0.000119492*pow(compression,4) - 0.027802576*pow(compression,3) + 2.423869094*pow(compression,2) - 93.736255939*compression + 1357.202193038;
	else if(compression < 74.8286)
		threshold = 0.8706*pow(compression,3) - 186.71*pow(compression,2) + 13348*compression - 318066;
	else if(compression < 90.20755)
		threshold = 4E-13 * pow(compression,7.5976);
	else
		threshold = 24.860132*compression- 1956.0132;
#endif

	if(CUDA)
	{
		unsigned char* CUinput;
		int* CUtransformed;
		int memSize = img->byteCount();

		cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
		cutilSafeCall(cudaMalloc((void**)&CUtransformed, sizeof(int) * width * height * 4));

		cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
		CUfwt97_2D_rgba(CUtransformed, CUinput, height, width);

		if(threshold)
			CUzeroOut(CUtransformed, threshold,  height*width*4);

		cutilSafeCall(cudaMemcpy(compressed, CUtransformed, sizeof(int)*height*width*4, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaFree(CUinput));
		cutilSafeCall(cudaFree(CUtransformed));
	}
	else
	{
		int i;
		int memsize = sizeof(float)*height*width*4;

		float* input = (float*) malloc(memsize);
		float* tempbank = (float*) malloc(sizeof(float)*(height>width?height:width));
		shuffleCPU(img->bits(), input, height, width);

		float* inputR = input;
		float* inputG = &input[height*width];
		float* inputB = &input[height*width*2];
		float* inputA = &input[height*width*3];

		for(i =0; i < height; i++){
			fwt97(&inputR[i*width], tempbank, width);
			fwt97(&inputG[i*width], tempbank, width);
			fwt97(&inputB[i*width], tempbank, width);
			fwt97(&inputA[i*width], tempbank, width);
		}

		transposeInPlace(inputR, height, width);
		transposeInPlace(inputG, height, width);
		transposeInPlace(inputB, height, width);
		transposeInPlace(inputA, height, width);

		for(i =0; i < width; i++){
			fwt97(&inputR[i*height], tempbank, height);
			fwt97(&inputG[i*height], tempbank, height);
			fwt97(&inputB[i*height], tempbank, height);
			fwt97(&inputA[i*height], tempbank, height);
		}

#ifdef TWODFWT
		transposeInPlace(inputR, width, height);
		transposeInPlace(inputG, width, height);
		transposeInPlace(inputB, width, height);
		transposeInPlace(inputA, width, height);

		for(i =0; i < height; i++){
			fwt97(&inputR[i*width], tempbank, width);
			fwt97(&inputG[i*width], tempbank, width);
			fwt97(&inputB[i*width], tempbank, width);
			fwt97(&inputA[i*width], tempbank, width);
		}

		transposeInPlace(inputR, height, width);
		transposeInPlace(inputG, height, width);
		transposeInPlace(inputB, height, width);
		transposeInPlace(inputA, height, width);

		for(i =0; i < width; i++){
			fwt97(&inputR[i*height], tempbank, height);
			fwt97(&inputG[i*height], tempbank, height);
			fwt97(&inputB[i*height], tempbank, height);
			fwt97(&inputA[i*height], tempbank, height);
		}
#endif
		zeroOut(input, threshold, height, width);
		RoundArray(compressed, input, width*height*4);

		free(tempbank);
		free(input);
	}
	return compressed;
}

void Decoder::decompress_image(QImage* img, int* compressed, bool CUDA)
{
	int width = img->width();
	int height = img->height();

	if(CUDA)
	{
		int* CUtransformed;
		unsigned char* CUoutput;
		cutilSafeCall(cudaMalloc((void**)&CUtransformed, sizeof(int) * width * height * 4));
		cutilSafeCall(cudaMalloc((void**)&CUoutput, sizeof(unsigned char) * width * height * 4));
		cutilSafeCall(cudaMemcpy(CUtransformed, compressed, sizeof(int)*width*height*4, cudaMemcpyHostToDevice));

		CUiwt97_2D_rgba(CUoutput, CUtransformed, height, width);

		cutilSafeCall(cudaMemcpy(img->bits(), CUoutput, img->byteCount(), cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaFree(CUtransformed));
		cutilSafeCall(cudaFree(CUoutput));
	}
	else
	{
		int i;
		float* tempbank = (float*)malloc(sizeof(float)*(height>width?height:width));
		float* output = (float*)malloc(sizeof(float)*4*width*height);
		intToFloat(output, compressed, width*height*4);

		float* outputR = output;
		float* outputG = &output[height*width];
		float* outputB = &output[height*width*2];
		float* outputA = &output[height*width*3];

		for(i =0; i < width; i++){
			iwt97(&outputR[i*height], tempbank, height);
			iwt97(&outputG[i*height], tempbank, height);
			iwt97(&outputB[i*height], tempbank, height);
			iwt97(&outputA[i*height], tempbank, height);
		}

		transposeInPlace(outputR, width, height);
		transposeInPlace(outputG, width, height);
		transposeInPlace(outputB, width, height);
		transposeInPlace(outputA, width, height);

		for(i =0; i < height; i++){
			iwt97(&outputR[i*width], tempbank, width);
			iwt97(&outputG[i*width], tempbank, width);
			iwt97(&outputB[i*width], tempbank, width);
			iwt97(&outputA[i*width], tempbank, width);
		}
#ifdef TWODFWT
		transposeInPlace(outputR, height, width);
		transposeInPlace(outputG, height, width);
		transposeInPlace(outputB, height, width);
		transposeInPlace(outputA, height, width);

		for(i =0; i < width; i++){
			iwt97(&outputR[i*height], tempbank, height);
			iwt97(&outputG[i*height], tempbank, height);
			iwt97(&outputB[i*height], tempbank, height);
			iwt97(&outputA[i*height], tempbank, height);
		}

		transposeInPlace(outputR, width, height);
		transposeInPlace(outputG, width, height);
		transposeInPlace(outputB, width, height);
		transposeInPlace(outputA, width, height);

		for(i =0; i < height; i++){
			iwt97(&outputR[i*width], tempbank, width);
			iwt97(&outputG[i*width], tempbank, width);
			iwt97(&outputB[i*width], tempbank, width);
			iwt97(&outputA[i*width], tempbank, width);
		}
#endif
		unshuffleCPU(output, img->bits(), height, width);

		free(output);
		free(tempbank);
	}
}

QImage* Encoder::compress_image_preview(QImage* img, float factor, double* psnr, bool CUDA)
{
	unsigned long numBytes;
	int* compressed = compress_image(img, factor, CUDA, &numBytes);

	QImage* decompressed = new QImage(img->width(), img->height(), QImage::Format_RGB32);
	Decoder::decompress_image(decompressed, compressed, CUDA);

	*psnr = Utility::psnr(img->bits(), decompressed->bits(), img->byteCount());

#ifdef DEBUGPRINT
	int i, zeroCoeff=0;
	for(i=0; i < img->width()*img->height()*4; i++){
		if(compressed[i]==0)
			zeroCoeff++;
	}
	float pct = 100*(zeroCoeff)/(float)(img->width()*img->height()*4);

	printf("%f\t%f\t%f\t",factor, pct, *psnr);
#endif

	free(compressed);
	return decompressed;
}

/** prevImg - the bits (img->bits()) of the previous frame
 *  currImg - the bits (img->bits()) of the current frame
 *  xOffset - the x index of the top-left pixel in the block. Should be 0,8,...width-8
 *  yOffset - the y index of the top-left pixel in the block. Should be 0,8,...height-8
 *  height - the height of prevImg, currImg, and diffImg
 *  width - the width of prevImg, currImg, and diffImg
 */
mvec motionVec8x8(unsigned char* prevImg, unsigned char* currImg, int xOffset, int yOffset, int height, int width)
{
	xOffset = xOffset*4; // for rgba
	int i, j, k, l, xIndex, yIndex;
	unsigned int diff;
	mvec vec;
	vec.diff = INT_MAX;
	for(i=-32; i < 36; i+=4){
		for(j=-8; j < 9; j++){
			diff=0;
			for(k=0; k < 32; k++){
				for(l=0; l < 8; l++){
					xIndex = xOffset + i + k;
					yIndex = yOffset + j + l;
					if(xIndex < 0 || xIndex >= width*4 || yIndex < 0 || yIndex >= height)
						diff +=     currImg[xOffset+k + (yOffset+l)*width*4];
					else
						diff += abs(currImg[xOffset+k + (yOffset+l)*width*4] - prevImg[xIndex + yIndex * width*4]);
				}
			}

			if(diff < vec.diff) {
				vec.diff = diff;
				vec.x = i;
				vec.y = j;
			}
			else if(diff == vec.diff) {
				if(sqrt(i*i + j*j) < sqrt(vec.x*vec.x + vec.y*vec.y)) {
					vec.x = i;
					vec.y = j;
				}
			}
		}
	}
	return vec;
}

mvec* motVecFrame(unsigned char* prevImg, unsigned char* currImg, int height, int width)
{
	int blockDimX = CEIL(width/8.0f);
	int blockDimY = CEIL(height/8.0f);
	mvec* vecs = (mvec*)malloc(sizeof(mvec) * blockDimX * blockDimY );

	for(int i=0; i < blockDimY; i++)
		for(int j=0; j< blockDimX; j++)
			vecs[j + i*blockDimX] = motionVec8x8(prevImg, currImg, 8*j, 8*i, height, width);

	return vecs;
}

int** Encoder::compress_video(QImage** original, int start_frame, int end_frame, mvec*** vecArrP, float compression, bool CUDA)
{
	int Qlevel = -5.12*compression + 512;
	int height = original[0]->height();
	int width = original[0]->width();
	int frames = end_frame-start_frame+1;

	*vecArrP = (mvec**)malloc(sizeof(mvec*)*frames);
	mvec** vecArr = *vecArrP;

	int** diff = (int**)malloc(sizeof(int*)*frames);
	for(int f = 0; f < frames; f++)
		diff[f] = (int*)malloc(sizeof(int)*width*height*4);

	short int* d = (short int*)malloc(sizeof(short int)*width*height*4);
	short int* dHat = (short int*)malloc(sizeof(short int)*width*height*4);
	short int* xHatPrev = (short int*)malloc(sizeof(short int)*width*height*4);

	for(int j=0; j < height; j++) {
		for (int k=0; k < width*4; k++) {
			xHatPrev[j+k*height] = 0;
		}
	}

	int xVec;
	int yVec;
	for(int i=0; i < frames; i++){
		unsigned char* frame_bits = original[i + start_frame]->bits();

		if(i==0){
			Utility u;
			vecArr[0] = u.zeroMvec(CEIL(width/8), CEIL(height/8));
		}
		else{
			if(CUDA)
				vecArr[i] = CUmotVecFrame(original[i + start_frame-1]->bits(), frame_bits, height, width);
			else
				vecArr[i] =   motVecFrame(original[i + start_frame-1]->bits(), frame_bits, height, width);
		}

#ifdef DEBUGPRINT
		if(i==2) // Print file for Matlab motion vector analysis. Only works for qcif video, always pulls the third frame. rm won't work on non-unix systems
		{
			FILE* secondFrame = fopen("secondFrame.txt","w");
			for(int y = 0; y < CEIL(height/8); y++){
				for(int z=0; z < CEIL(width/8); z++){
					fprintf(secondFrame, "%d\t%d\t%d\n",vecArr[i][z + y * CEIL(width/8)].x/4 , vecArr[i][z + y * CEIL(width/8)].y , vecArr[i][z + y * CEIL(width/8)].diff);
				}
			}

			fclose(secondFrame);
			Utility u;
			u.MfileConverter("secondFrame.txt", "vectors.m");
			system("rm secondFrame.txt");
		}
#endif
		for(int j=0; j < height; j++){
			for (int k=0; k < width*4; k++){
				xVec = vecArr[i][(int)j/8 + (int)k/32 * CEIL(height/8)].x;
				yVec = vecArr[i][(int)j/8 + (int)k/32 * CEIL(height/8)].y;
				if((j+yVec) < 0 ||(j+yVec) >= height ||(k+xVec) < 0 ||(k+xVec)/4 >=width)
					diff [i][k + j*width*4] = floor(((frame_bits[k + j*width*4])/(float)NUM_SYMBOLS)*Qlevel);
				else
					diff [i][k + j*width*4] = floor(((frame_bits[k + j*width*4] - xHatPrev[(k+xVec) + (j+yVec)*width*4])/(float)NUM_SYMBOLS)*Qlevel);
				d       [k + j*width*4] = frame_bits[k + j*width*4] - xHatPrev[k + j*width*4];
				dHat    [k + j*width*4] = round( floor((d[k + j*width*4]/(float)NUM_SYMBOLS)*Qlevel) * (NUM_SYMBOLS/(float)Qlevel));
				xHatPrev[k + j*width*4] = dHat[k + j*width*4] + xHatPrev[k + j*width*4];
			}
		}
	}

	free(d);
	free(dHat);
	free(xHatPrev);

	return diff;
}

QImage** Decoder::decompress_video(int** diff, int frames, mvec** vecArr, float compression, int height, int width)
{
	int Qlevel = -5.12*compression + 512;
	/* might be leaing memory on output mallocs */
	QImage** output = (QImage**) malloc(frames * sizeof(QImage*));
	for(int f = 0; f < frames; f++)
		output[f] = new QImage(width, height, QImage::Format_ARGB32);

	unsigned char* prevFrame = (unsigned char*)malloc(sizeof(unsigned char)*height*width*4);
	for(int i = 0; i < width*height*4; i++)
		prevFrame[i] = 0;

	int xVec;
	int yVec;

	for(int i=0; i < frames; i++){
		for(int j=0; j < height; j++){
			for (int k=0; k < width*4; k++){
				xVec = vecArr[i][((int)j/8 + (int)k/32 * CEIL(height/8))].x;
				yVec = vecArr[i][((int)j/8 + (int)k/32 * CEIL(height/8))].y;


				if((j+yVec) < 0 ||(j+yVec) >= height ||(k+xVec) < 0 ||(k+xVec)/4 >=width)
					prevFrame[k + j*width*4] = CLAMP(round((diff[i][k + j*width*4] / (float)Qlevel) *NUM_SYMBOLS));
				else
					prevFrame[k + j*width*4] = CLAMP(prevFrame[(k+xVec) + (j+yVec)*width*4] + round((diff[i][k + j*width*4] / (float)Qlevel) *NUM_SYMBOLS));
			}
		}
		memcpy(output[i]->bits(), prevFrame, output[i]->byteCount());
	}

	free(prevFrame);
	return output;
}

QImage** Encoder::compress_video_preview(QImage** original, int start_frame, int end_frame, float compression, double* psnr, bool CUDA, int* time)
{
	QTime timer;

	int frames = end_frame - start_frame +1;
	mvec** vec;

	timer.restart();
	int** comp = compress_video(original, start_frame, end_frame, &vec, compression, CUDA);
	*time = timer.elapsed();
	QImage** output = Decoder::decompress_video(comp, frames, vec, compression, original[0]->height(), original[0]->width());

	for(int f = 0; f < frames; f++)
	{
		free(vec[f]);
		free(comp[f]);
	}
	free(comp);
	free(vec);

	*psnr = Utility::psnr_video(original, output, frames);

#ifdef DEBUGPRINT
	double pctZeros = Utility::pct_zeros(comp, frames, original[0]->height()*original[0]->width()*4);
	printf("%f\t%f\t%f\t\n", pctZeros, *time, *psnr);
	fflush(stdout);
#endif

	return output;
}
