#include "mainwindow.h"

QImage* MainWindow::grayscale_image()
{
	QImage* img = this->display->getRightImage();
	int r = 0, g = 0, b = 0;

	// Scale each RGB value by the brightening factor
	for(int y = 0; y < img->height(); y++)
	{
		for(int x = 0; x < img->width(); x++)
		{
			QRgb p = img->pixel(x, y);
			r = g = b = (qRed(p) * 0.3) + (qGreen(p) * 0.59) + (qBlue(p) * 0.11);
			img->setPixel(x, y, qRgb(r, g, b));
		}
	}
	return img;
}

QImage* MainWindow::grayscale_video()
{
	return NULL;
}
