#include "mainwindow.h"
#include "dwt97.h"

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

int* Encoder::compress_image(QImage* img, float factor, bool CUDA)
{
	int width = img->width();
	int height = img->height();
	int* compressed = (int*)malloc(sizeof(int)*height*width*4);

	float threshold;
	if(factor < 50)
		threshold = 0;
	else if(factor < 77.703568)
		threshold = 0.000116*pow(factor,3) - 0.019118*pow(factor,2) + 1.108995*factor - 21.184998;
	else if(factor < 91.641708)
		/*PROBLEM*/
		threshold = 0.000046386058562*pow(factor,6)-0.023471363133066*pow(factor,5)+4.94695855894703*pow(factor,4)-555.891027747718*pow(factor,3)+35124.430525745*pow(factor,2)-1183223.56408731*factor+16601453.5673362;
	else if(factor < 92.569443)
		threshold = 0.873366*factor + 12.532677;
	else if(factor < 94.001099)
		threshold = 153.426432*pow(factor,4) - 57142.000968*pow(factor,3) + 7980724.953524*pow(factor,2) - 495389786.739970*factor + 11531433455.627500;
	else if(factor < 98.455238)
		threshold = 0.0000000000320307431396312 * pow(E,0.315611691031623*factor);
	else
		threshold = 1.053774*factor - 3.773881;

//	printf("%f\t",threshold);



/*	if(factor < 19.4963201)
		threshold = 0;
	else if(factor <=52.7)
		threshold = factor*0.045110 - 0.879479;
	else if(factor < 68.29137)
		threshold = 0.000119492*pow(factor,4) - 0.027802576*pow(factor,3) + 2.423869094*pow(factor,2) - 93.736255939*factor + 1357.202193038;
	else if(factor < 74.8286)
		threshold = 0.8706*pow(factor,3) - 186.71*pow(factor,2) + 13348*factor - 318066;
	else if(factor < 90.20755)
		threshold = 4E-13 * pow(factor,7.5976);
	else
		threshold = 24.860132*factor- 1956.0132;
*/
	if(CUDA)
	{
		unsigned char* CUinput;
		int* CUtransformed;
		int memSize = img->byteCount();

		cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
		cutilSafeCall(cudaMalloc((void**)&CUtransformed, sizeof(int) * width * height * 4));

		cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
		CUfwt97_2D_rgba(CUtransformed, CUinput, height, width);

		CUzeroOut(CUtransformed, threshold,  height*width*4);

		cutilSafeCall(cudaMemcpy(compressed, CUtransformed, sizeof(int)*height*width*4, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaFree(CUinput));
		cutilSafeCall(cudaFree(CUtransformed));
	}
	else
	{
		int i;
		int memsize = sizeof(float)*height*width*4;

		float* input = (float*)malloc(memsize);
		float* tempbank = (float*)malloc(sizeof(float)*(height>width?height:width));
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

		unshuffleCPU(output, img->bits(), height, width);

		free(output);
		free(tempbank);
	}
}

QImage* Encoder::compress_image_preview(QImage* img, float factor, double* psnr, bool CUDA)
{
	int* compressed = compress_image(img, factor, CUDA);

	int i, zeroCoeff=0;
	for(i=0; i < img->width()*img->height()*4; i++){
		if(compressed[i]==0)
			zeroCoeff++;
	}
//	float pct = 100*(zeroCoeff)/(float)(img->width()*img->height()*4);

	QImage* decompressed = new QImage(img->width(), img->height(), QImage::Format_RGB32);
	Decoder::decompress_image(decompressed, compressed, CUDA);
	free(compressed);

	*psnr = Utility::psnr(img->bits(), decompressed->bits(), img->byteCount());

//	printf("%2.6f\t%2.6f\t%2.6f\n",factor, pct, *psnr);
//	fflush(stdout);

	return decompressed;
}

/** prevImg - the bits (img->bits()) of the previous frame
 *  currImg - the bits (img->bits()) of the current frame
 *  diffImg - the bits (img->bits()) of the difference of the frames
 *  xOffset - the x index of the top-left pixel in the block. Should be 0,8,...width-8
 *  yOffset - the y index of the top-left pixel in the block. Should be 0,8,...height-8
 *  height - the height of prevImg, currImg, and diffImg
 *  width - the width of prevImg, currImg, and diffImg
 */
int* motionVec8x8(unsigned char* prevImg, unsigned char* currImg, int* diffBlock, int xOffset, int yOffset, int height, int width)
{
	xOffset = xOffset*4; // for rgba
	int i, j, k, l, m, xIndex, yIndex, diff, minDiff= INT_MAX;
	int* vec = (int*)malloc(sizeof(int)*2);
	for(i=-7; i < 8; i++){
		i*=4;
		for(j=-7; j < 8; j++){
			diff=0;
			for(k=0; k < 8; k++){
				for(l=0; l < 8; l++){
					for(m = 0; m < 4; m++){
						xIndex = xOffset + i + k + m;
						yIndex = yOffset + j + l;
						if(xIndex < 0 || xIndex > width || yIndex < 0 || yIndex > height)
							diff += currImg[xIndex + yIndex * width];
						else
							diff += abs(currImg[xIndex+m + yIndex * width] - prevImg[xOffset+k + m + (yOffset+l)*width]);
					}
				}
			}
			if(diff < minDiff){
				i/=4;
				minDiff = diff;
				vec[0] = i;
				vec[1] = j;
				i*=4;
			}
		}
		i/=4;
	}
	for(i=0; i < 8; i++){
		for(j=0; j < 8; j++){
			for(k=0; k < 4; k++){
				diffBlock[xOffset+i+k + (yOffset+j)*width] = currImg[xOffset + vec[0] + i + k + (yOffset + vec[1] + j)*width] - prevImg[xOffset+i + k + (yOffset+j)*width];
			}
		}
	}
	return vec;
}

#define NUM_SYMBOLS 512 // -256 -> 255

int** Encoder::compress_video(QImage** original, int frames, int* vecArr, int Qlevel)
{
	int height = original[0]->height();
	int width = original[0]->width();

	int** diff = (int**)malloc(sizeof(int*)*frames);
	for(int f = 0; f < frames; f++)
		diff[f] = (int*)malloc(sizeof(int)*width*height*4);

	short int* d = (short int*)malloc(sizeof(short int)*width*height*4);
	short int* dHat = (short int*)malloc(sizeof(short int)*width*height*4);
	short int* xHatPrev = (short int*)malloc(sizeof(short int)*width*height*4);

	for(int j=0; j < height; j++){
		for (int k=0; k < width*4; k++){
			xHatPrev[j+k*height]=0;
		}
	}

	for(int i=0; i < frames; i++){
		for(int j=0; j < height; j++){
			for (int k=0; k < width*4; k++){
				d       [j + k*height] = original[i]->bits()[j + k*height] - xHatPrev[j + k*height];
				dHat    [j + k*height] = round( floor((d[j + k*height]/(float)NUM_SYMBOLS)*Qlevel) * (NUM_SYMBOLS/(float)Qlevel));
				xHatPrev[j + k*height] = dHat[j + k*height] + xHatPrev[j + k*height];
				diff [i][j + k*height] = floor((d[j + k*height]/(float)NUM_SYMBOLS)*Qlevel);
			}
		}
	}

	free(d);
	free(dHat);
	free(xHatPrev);

	return diff;
}

QImage** Decoder::decompress_video(int** diff, int frames, int* vecArr, int Qlevel, int height, int width)
{
	QImage** output = (QImage**) malloc(frames * sizeof(QImage*));
	for(int f = 0; f < frames; f++){
		output[f] = new QImage(width, height, QImage::Format_ARGB32);
	}

	unsigned char* prevFrame = (unsigned char*)malloc(sizeof(unsigned char)*height*width*4);

	for(int i=0; i < width*height*4; i++){
		prevFrame[i]=0;
	}

	for(int i=0; i < frames; i++){
		for(int j=0; j < height; j++){
			for (int k=0; k < width*4; k++){
				prevFrame[j+k*height] = CLAMP(prevFrame[j+k*height] + round((diff[i][j+k*height] / (float)Qlevel) *NUM_SYMBOLS));
			}
		}
		memcpy(output[i]->bits(), prevFrame, output[i]->byteCount());
	}

	free(prevFrame);
	return output;
}

QImage** Encoder::compress_video_preview(QImage** original, int frames, int Qlevel, double* psnr)
{
	int* vec;

	int** comp = compress_video(original, frames, vec, Qlevel);
	QImage** output = Decoder::decompress_video(comp, frames, vec, Qlevel, original[0]->height(), original[0]->width());

	for(int f = 0; f < frames; f++){
		free(comp[f]);
	}
	free(comp);
	free(vec);

	*psnr = Utility::psnr_video(original, output, frames);
//	printf("%d\t%7.4f\n",Qlevel, *psnr);
//	fflush(stdout);

	return output;
}
