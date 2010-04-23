#include "mainwindow.h"

QImage* MainWindow::compress_image(float factor)
{
    QImage* img = this->display->getRightImage();
    int width = img->width();
    int height = img->height();

    if(CUDA_CAPABLE && CUDA_ENABLED)
    {
		unsigned char* CUinput;
		float* CUtransformed;
		unsigned char* CUoutput;
		int memSize = img->byteCount();

		cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
		cutilSafeCall(cudaMalloc((void**)&CUtransformed, sizeof(float) * width * height * 4));
		cutilSafeCall(cudaMalloc((void**)&CUoutput, memSize));

		cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
		CUfwt97_2D(CUtransformed, CUinput, height, width * 4);
		CUzeroOut(CUtransformed, factor,  height*width*4);

		float* transformed = (float*)malloc(sizeof(float)*width*height*4);
		cutilSafeCall(cudaMemcpy(transformed, CUtransformed, sizeof(float)*width*height*4, cudaMemcpyDeviceToHost));
		FILE* trans = fopen("transformed.csv", "w");
		int i;
		for(i=0; i < width*height*4; i++){
			fprintf(trans, "%f\t",transformed[i]);
			if(!(i%5))
				fprintf(trans, "\n");
		}
		fclose(trans);
		free(transformed);

//		CUquantize(CUtransformed, factor, 255, width*height*4);

		CUiwt97_2D(CUoutput, CUtransformed, height, width*4);
		cutilSafeCall(cudaMemcpy(img->bits(), CUoutput, memSize, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaFree(CUinput));
		cutilSafeCall(cudaFree(CUtransformed));
		cutilSafeCall(cudaFree(CUoutput));

		return img;
    }
    else
    {
		return img;
    }
}

QImage* MainWindow::compress_video(float factor)
{
    QImage* img = this->display->getRightImage();
    int width = img->width();
    int height = img->height();

    if(CUDA_CAPABLE && CUDA_ENABLED)
    {
		return img;
    }
    else
    {
		return img;
    }
}
