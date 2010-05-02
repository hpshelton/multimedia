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

QImage** MainWindow::compress_video(float factor)
{
	QImage** original = this->video_display->getRightVideo();
	QImage** modified = (QImage**) malloc(this->frames * sizeof(QImage*));
	for(int f = 0; f < this->frames; f++)
		modified[f] = compress_preview(new QImage(*original[f]), factor);
	return modified;
}
