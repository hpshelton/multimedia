#include "mainwindow.h"

QImage* MainWindow::brighten_image(float factor)
{
	QImage* img = this->display->getRightImage();
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
		CUbrighten(CUoutput, CUinput, height, width, factor);
		cutilSafeCall(cudaMemcpy(img->bits(), CUoutput, memSize, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaFree(CUinput));
		cutilSafeCall(cudaFree(CUoutput));

		return img;
	}
	else
	{
		// Scale each RGB value by the brightening factor
		for(int y = 0; y < img->height(); y++)
		{
			for(int x = 0; x < img->width(); x++)
			{
				QRgb p = img->pixel(x, y);
				int r = qRed(p) * factor;
				r = CLAMP(r);
				int g = qGreen(p) * factor;
				g = CLAMP(g);
				int b = qBlue(p) * factor;
				b = CLAMP(b);
				img->setPixel(x, y, qRgb(r, g, b));
			}
		}
		return img;
	}
}

QImage* MainWindow::brighten_video(float factor)
{
	return NULL;
}
