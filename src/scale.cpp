#include "mainwindow.h"

QImage* MainWindow::scale_image(QImage* img, float factor)
{
	int width = img->width();
	int height = img->height();
	int newWidth = width * factor;
	int newHeight = height * factor;
	float factorInverse = 1/factor;

	QImage* newImg = new QImage(newWidth, newHeight, QImage::Format_RGB32);

	// Scale the image
	for(int x = 0; x < newWidth; x++)
	{
		for(int y = 0; y < newHeight; y++)
		{
			float u = factorInverse*x;
			float v = factorInverse*y;
			newImg->setPixel(x, y, Utility::GaussianSample(img, u, v, 0.6, 2*factor));
		}
	}
	return newImg;
}

QImage** MainWindow::scale_video(float factor)
{
	QImage** original = this->video_display->getRightVideo();
	QImage** modified = (QImage**) malloc(this->frames * sizeof(QImage*));
	for(int f = 0; f < this->frames; f++)
		 modified[f] = scale_image(new QImage(*original[f]), factor);
	return modified;
}
