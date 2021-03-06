#include "mainwindow.h"

QImage* MainWindow::grayscale_image(QImage* img)
{
	int width = img->width();
	int height = img->height();

	if(CUDA_CAPABLE && CUDA_ENABLED)
	{
		unsigned char* CUinput;
		unsigned char* CUoutput;
		int memSize = img->byteCount();

		cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
		cutilSafeCall(cudaMalloc((void**)&CUoutput, memSize));

		cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
		CUgreyscale(CUoutput, CUinput, height, width);
		cutilSafeCall(cudaMemcpy(img->bits(), CUoutput, memSize, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaFree(CUinput));
		cutilSafeCall(cudaFree(CUoutput));
	}
	else
	{
		int r = 0, g = 0, b = 0;

		// Convert each RGB to grayscale
		for(int y = 0; y < img->height(); y++)
		{
			for(int x = 0; x < img->width(); x++)
			{
				QRgb p = img->pixel(x, y);
				r = g = b = (qRed(p) * 0.3) + (qGreen(p) * 0.59) + (qBlue(p) * 0.11);
				img->setPixel(x, y, qRgb(r, g, b));
			}
		}
	}
	return img;
}

QImage** MainWindow::grayscale_video()
{
	QImage** original = this->video_display->getRightVideo();
	QImage** modified = (QImage**) malloc(this->frames * sizeof(QImage*));
	for(int f = 0; f < this->frames; f++)
		 modified[f] = grayscale_image(new QImage(*original[f]));
	return modified;
}
