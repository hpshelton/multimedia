#include "mainwindow.h"

QImage* MainWindow::crop_image(QImage* img, int x1, int x2, int y1, int y2)
{
	int newWidth = x2 - x1;
	int newHeight = y2 - y1;

	// Instantiate the output image
	QImage* newImg = new QImage(newWidth, newHeight, QImage::Format_RGB32);

	// Crop the original image
	for(int x = x1; x < x2; x++)
		for(int y = y1; y < y2; y++)
			newImg->setPixel(x-x1, y-y1, img->pixel(x,y));
	return newImg;
}

QImage** MainWindow::crop_video(int x1, int x2, int y1, int y2)
{
	QImage** original = this->video_display->getRightVideo();
	QImage** modified = (QImage**) malloc(this->frames * sizeof(QImage*));
	for(int f = 0; f < this->frames; f++)
		 modified[f] = crop_image(new QImage(*original[f]), x1, x2, y1, y2);
	return modified;
}

