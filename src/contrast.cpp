#include "mainwindow.h"

QImage* MainWindow::contrast_image(float factor)
{
	QImage* img = this->image_display->getRightImage();

	int width = img->width();
	int height = img->height();

	if(CUDA_CAPABLE && CUDA_ENABLED) {
		return img;
	}
	else{
		int r = 0, g = 0, b = 0;

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
		return img;
	}
}

QImage** MainWindow::contrast_video(float factor)
{
	return NULL;
}
