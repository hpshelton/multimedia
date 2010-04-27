#include "mainwindow.h"

QImage* MainWindow::crop_image(int x1, int x2, int y1, int y2)
{
	QImage* img = this->image_display->getRightImage();

	//Validate input
	if(x1 < 0 || x2 < 0 || y1 < 0 || y2 < 0)
	{
		fprintf(stderr,"Improper pixel coordinates\n");
		return img;
	}

	int rightX = x2;
	int leftX = x1;
	int rightY = y2;
	int leftY = y1;

	// Swap out-of-order input
	if(rightX < leftX)
	{
		leftX = x2;
		rightX = x1;
	}
	if(rightY < leftY)
	{
		leftY = y2;
		rightY = y1;
	}

	int newWidth = rightX-leftX;
	int newHeight = rightY-leftY;

	// Validate the target dimensions
	if(newWidth > img->width())
	{
		fprintf(stderr,"Improper width - %d pixels is larger than original image\n", newWidth);
		return img;
	}
	if(newWidth == 0)
	{
		fprintf(stderr,"Improper width - Cannot crop image to %d pixels\n", newWidth);
		return img;
	}
	if(newHeight > img->height())
	{
		fprintf(stderr,"Improper height - %d pixels is larger than original image\n", newHeight);
		return img;
	}
	if(newHeight == 0)
	{
		fprintf(stderr,"Improper height - Cannot crop image to %d pixels\n", newHeight);
		return img;
	}

	// Instantiate the output image
	QImage* newImg = new QImage(newWidth, newHeight, QImage::Format_RGB32);

	// Crop the original image
	for(int x = leftX; x < rightX; x++)
	{
		for(int y = leftY; y < rightY; y++)
			newImg->setPixel(x-leftX, y-leftY, img->pixel(x,y));
	}
	return newImg;
}

QImage** MainWindow::crop_video(int x1, int x2, int y1, int y2)
{
	return NULL;
}
