#include "mainwindow.h"

QImage* MainWindow::brighten_image(float factor)
{
	QImage* img = this->display->getRightImage();

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

QImage* MainWindow::brighten_video(float factor)
{
	return NULL;
}
