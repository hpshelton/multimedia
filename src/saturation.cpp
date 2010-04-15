#include "mainwindow.h"

QImage* MainWindow::saturate_image(float factor)
{
	QImage* img = this->file[0];
	int width = img->width();
	int height = img->height();

	if(CUDA_CAPABLE && CUDA_ENABLED) {
		unsigned char* CUinput;
		unsigned char* CUoutput;
		int memSize = img->byteCount();

		cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
		cutilSafeCall(cudaMalloc((void**)&CUoutput, memSize));

		cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
		CUsaturate(CUoutput, CUinput, height, width, factor);
		cutilSafeCall(cudaMemcpy(img->bits(), CUoutput, memSize, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaFree(CUinput));
		cutilSafeCall(cudaFree(CUoutput));

		return img;
	}
	else{
		int r = 0, g = 0, b = 0;
		float lum = 0;

		// Modify the saturation
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				QRgb p = img->pixel(x, y);
				lum = (qRed(p) * 0.3) + (qGreen(p) * 0.59) + (qBlue(p) * 0.11);

				r = (1-factor)*lum + factor*(qRed(p));
				r = CLAMP(r);

				g = (1-factor)*lum + factor*(qGreen(p));
				g = CLAMP(g);

				b = (1-factor)*lum + factor*(qBlue(p));
				b = CLAMP(b);

				img->setPixel(x,y, qRgb(r, g, b));
			}
		}
		return img;
	}
}

QImage* MainWindow::saturate_video(float factor)
{
	return NULL;
}

