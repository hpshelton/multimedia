#include "mainwindow.h"
#include <iostream>

QImage* MainWindow::brighten_image(float factor)
{
	QImage* img = this->file[0];

	// Scale each RGB value by the brightening factor
	for(int y = 0; y < img->height(); y++)
	{
		for(int x = 0; x < img->width(); x++)
		{
			QRgb p = img->pixel(x, y);
			int r = CLAMP(qRed(p) * factor);
			int g = CLAMP(qGreen(p) * factor);
			int b = CLAMP(qBlue(p) * factor);
			img->setPixel(x, y, qRgb(r, g, b));
		}
	}
	return img;
}

QImage* MainWindow::brighten_video(float factor)
{
	return NULL;
}
