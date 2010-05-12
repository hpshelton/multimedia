#include "mainwindow.h"

QImage* MainWindow::contrast_image(QImage* img, float factor)
{
	int width = img->width();
	int height = img->height();

	// Calculate the average luminosity
	float lum = 0;
	for(int y = 0; y < height; y++)
	{
		for(int x = 0; x < width; x++)
		{
			QRgb p = img->pixel(x, y);
			lum += (qRed(p) * 0.3) + (qGreen(p) * 0.59) + (qBlue(p) * 0.11);
		}
	}
	lum /= width*height;

	if(CUDA_CAPABLE && CUDA_ENABLED)
	{
		unsigned char* CUinput;
		unsigned char* CUoutput;
		int memSize = img->byteCount();

		cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
		cutilSafeCall(cudaMalloc((void**)&CUoutput, memSize));

		cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
		CUcontrast(CUoutput, CUinput, height, width, factor, lum);
		cutilSafeCall(cudaMemcpy(img->bits(), CUoutput, memSize, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaFree(CUinput));
		cutilSafeCall(cudaFree(CUoutput));
	}
	else
	{
		float r = 0, g = 0, b = 0;

		// Modify the contrast
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				QRgb in = img->pixel(x, y);

				r = (1-factor)*lum + factor*(qRed(in));
				r = CLAMP(r);

				g = (1-factor)*lum + factor*(qGreen(in));
				g = CLAMP(g);

				b = (1-factor)*lum + factor*(qBlue(in));
				b = CLAMP(b);

				img->setPixel(x,y, qRgb(r, g, b));
			}
		}
	}
	return img;
}

QImage** MainWindow::contrast_video(float factor)
{
	QImage** original = this->video_display->getRightVideo();
	QImage** modified = (QImage**) malloc(this->frames * sizeof(QImage*));
	for(int f = 0; f < this->frames; f++)
		 modified[f] = contrast_image(new QImage(*original[f]), factor);
	return modified;
}

