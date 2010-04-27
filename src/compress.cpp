#include "mainwindow.h"

QImage* MainWindow::compress_image(float factor)
{
	 QImage* img = this->image_display->getRightImage();
	 int width = img->width();
	 int height = img->height();

	 if(CUDA_CAPABLE && CUDA_ENABLED)
	 {
		unsigned char* CUinput;
		float* CUtransformed;
		unsigned char* CUoutput;
		int memSize = img->byteCount();

		float threshold;
		if(factor < 19.5)
			threshold = 0;
		else if(factor <=50)
			threshold = factor*0.045110 - 0.879479;
		else if(factor < 70)
			threshold = 0.000119492*pow(factor,4) - 0.027802576*pow(factor,3) + 2.423869094*pow(factor,2) - 93.736255939*factor + 1357.268193043;
		else if(factor < 75)
			threshold = 0.8706*pow(factor,3) - 186.71*pow(factor,2) + 13348*factor - 318066;
		else if(factor < 90)
			threshold = 4E-13 * pow(factor,7.5976);
		else if(factor <= 99.99)
			threshold = 52.32*factor - 4433.1;
		else
			factor = 550;

		cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
		cutilSafeCall(cudaMalloc((void**)&CUtransformed, sizeof(float) * width * height * 4));
		cutilSafeCall(cudaMalloc((void**)&CUoutput, memSize));

		cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
		CUfwt97_2D_rgba(CUtransformed, CUinput, height, width);

		CUzeroOut(CUtransformed, threshold,  height*width*4);
//		CUquantize(CUtransformed, factor, 255, width*height*4);


/*		DEBUG TO SEE THE COEFF.
		float* transformed = (float*)malloc(sizeof(float)*width*height*4);
		cutilSafeCall(cudaMemcpy(transformed, CUtransformed, sizeof(float)*width*height*4, cudaMemcpyDeviceToHost));
		FILE* trans = fopen("transformed.csv", "w");
		int i;
		for(i=0; i < width*height*4; i++){
			fprintf(trans, "%f\t",transformed[i]);
			if(!(i%4))
				fprintf(trans, "\n");
		}
		fclose(trans);
		free(transformed);
*/
/*		DEBUG TO SEE PCT ZEROS
		float* transformed = (float*)malloc(sizeof(float)*width*height*4);
		cutilSafeCall(cudaMemcpy(transformed, CUtransformed, sizeof(float)*width*height*4, cudaMemcpyDeviceToHost));
		int i, zeroCoeff=0;
		for(i=0; i < width*height*4; i++){
			if(transformed[i]==0)
				zeroCoeff++;
		}
		printf("%f\t%f\t%f\n", 100*(zeroCoeff)/(float)(width*height*4),factor, threshold);
		fflush(stdout);
		free(transformed);
*/
		CUiwt97_2D_rgba(CUoutput, CUtransformed, height, width);
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
	 QImage* img = this->image_display->getRightImage();
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
