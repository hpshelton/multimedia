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

int* MainWindow::compress_image(QImage* img, float factor)
{
	int width = img->width();
	int height = img->height();
	int* compressed = (int*)malloc(sizeof(int)*height*width*4);

	float threshold;
	if(factor < 19.4963201)
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

	if(CUDA_CAPABLE && CUDA_ENABLED)
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
		zeroOut(input, threshold, height, width);
		RoundArray(compressed, input, width*height*4);
		free(tempbank);
		free(input);
	}
	return compressed;
}

void MainWindow::decompress_image(QImage* img, int* compressed)
{
	int width = img->width();
	int height = img->height();

	if(CUDA_CAPABLE && CUDA_ENABLED)
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

		unshuffleCPU(output, img->bits(), height, width);

		free(output);
		free(tempbank);
	}
}

QImage* MainWindow::compress_preview(QImage* img, float factor)
{
	int* compressed = compress_image(img, factor);
	decompress_image(img, compressed);
	free(compressed);
	return img;
}

#define CEIL(a) ( (a - (int)a)==0 ? (int)a : (int)a+1 )

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

/** Compresses the video by exhaustively finding motion estimation vectors
  * These vectors are returned in vecArr (allocated in the method)
  * even indices in vecArr are x components
  * odd indices in vecArr are y components
  * vector pairs correspond to 8x8 blocks, 1st frame, 1st row, 1st block then 1st frame, 1st row, 2nd block, etc
  * the method returns the residual frame difference, in int** form (allocated in the method)
  */
int** MainWindow::compress_video(int* vecArr)
{
	 if(CUDA_CAPABLE && CUDA_ENABLED)
	 {
		 int** null = 0;
		 return null;
//		return this->video_display->getRightVideo();
	 }
	 else
	 {
		QImage** original = this->video_display->getRightVideo();
		int height = original[0]->height();
		int width = original[0]->width();

		int index=0;
		int* vec;
		vecArr = (int*)malloc(sizeof(int) * this->frames * CEIL(height/8.0f) * CEIL(width/8.0f) * 2);

		int** diff = (int**)malloc(sizeof(int*)*this->frames);
		for(int i=0; i < frames; i++)
			diff[i] = (int*)malloc(sizeof(int)*width*height*4);

		for(int i=0; i < height*width*4; i++)
			diff[0][i] = original[0]->bits()[i];

		for(int frame=1; frame < this->frames; frames++){
			for(int j=0; j < height; j+=8){
				for(int i=0; i < width; i+=8){
					vec = motionVec8x8(original[frame]->bits(), original[frame-1]->bits(), diff[frame], i, j, height, width);
					vecArr[index++] = vec[0];
					vecArr[index++] = vec[1];
					free(vec);
				}
			}
		}
		return diff;
/*
		QImage** modified = (QImage**) malloc(this->frames * sizeof(QImage*));
		for(int f = 0; f < this->frames; f++)
			 modified[f] = compress_image(new QImage(*original[f]), factor);
		return modified;
*/	 }


}
