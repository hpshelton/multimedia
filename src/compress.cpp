#include "mainwindow.h"
#include "dwt97.h"

QImage* MainWindow::compress_image(float factor)
{
	;
}

QImage* MainWindow::decompress_image()
{
	;
}

QImage* MainWindow::compress_preview(QImage* img, float factor)
{
	 int width = img->width();
	 int height = img->height();

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
		float* CUtransformed;
		unsigned char* CUoutput;
		int memSize = img->byteCount();

		cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
		cutilSafeCall(cudaMalloc((void**)&CUtransformed, sizeof(float) * width * height * 4));
		cutilSafeCall(cudaMalloc((void**)&CUoutput, memSize));

		cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
		CUfwt97_2D_rgba(CUtransformed, CUinput, height, width);

		CUzeroOut(CUtransformed, threshold,  height*width*4);
//		CUquantize(CUtransformed, factor, 255, width*height*4);


//		DEBUG TO SEE THE COEFF.
//		float* transformed = (float*)malloc(sizeof(float)*width*height*4);
//		cutilSafeCall(cudaMemcpy(transformed, CUtransformed, sizeof(float)*width*height*4, cudaMemcpyDeviceToHost));
//		FILE* trans = fopen("transformed.csv", "w");
//		int i;
//		for(i=0; i < width*height*4; i++){
//                        fprintf(trans, "%10.6f\t",transformed[i]);
//			if(!(i%4))
//				fprintf(trans, "\n");
//		}
//		fclose(trans);
//		free(transformed);

//		DEBUG TO SEE PCT ZEROS
//		float* transformed = (float*)malloc(sizeof(float)*width*height*4);
//		cutilSafeCall(cudaMemcpy(transformed, CUtransformed, sizeof(float)*width*height*4, cudaMemcpyDeviceToHost));
//		int i, zeroCoeff=0;
//		for(i=0; i < width*height*4; i++){
//			if(transformed[i]==0)
//				zeroCoeff++;
//		}
//		printf("%f\t%f\t%f\n", 100*(zeroCoeff)/(float)(width*height*4),factor, threshold);
//		fflush(stdout);
//		free(transformed);

		CUiwt97_2D_rgba(CUoutput, CUtransformed, height, width);
		cutilSafeCall(cudaMemcpy(img->bits(), CUoutput, memSize, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaFree(CUinput));
		cutilSafeCall(cudaFree(CUtransformed));
		cutilSafeCall(cudaFree(CUoutput));

		return img;
	}
	 else
	 {
		 int i;
		 int memsize = sizeof(float)*height*width*4;

		 //////////////
		 // Compress //
		 //////////////

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

		 //////////////////
		 // Transmission //
		 //////////////////

		 float* output = (float*)malloc(memsize);
		 memcpy(output, input, memsize);
		 zeroOut(output, threshold, height, width);

		 ////////////////
		 // Decompress //
		 ////////////////

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
		 free(input);
		 free(tempbank);
		 return img;
	 }
}

QImage** MainWindow::compress_video(float factor)
{
	 if(CUDA_CAPABLE && CUDA_ENABLED)
	 {
		 return this->video_display->getRightVideo();
	 }
	 else
	 {
		 QImage** original = this->video_display->getRightVideo();
		 QImage** modified = (QImage**) malloc(this->frames * sizeof(QImage*));
		 for(int f = 0; f < this->frames; f++)
			  modified[f] = compress_preview(new QImage(*original[f]), factor);
		 return modified;
	 }


}
