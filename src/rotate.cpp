#include "mainwindow.h"

QImage* MainWindow::rotate_image(float a)
{
	QImage* img = this->image_display->getRightImage();
	int width = img->width();
	int height = img->height();
	int newHeight = 0;
	int newWidth = 0;

	// Determine new dimensions
	if(a <= 90)
	{
		a *= (PI/180.0);
		newHeight = ceil(height/cos(a)+sin(a)*(width-(height*tan(a))));
		newWidth = ceil(width/cos(a)+sin(a)*(height-(width*tan(a))));
	}
	else if(a <= 180)
	{
		a*= (PI/180.0);
		newHeight = ceil(abs(abs(height/cos(a)))+sin(a)*(width-(abs(height*tan(a)))));
		newWidth = ceil(abs(abs(width/cos(a)))+sin(a)*(height-(abs(width*tan(a)))));
	}
	else if(a <= 270)
	{
		a = (a*(PI/180.0)) - PI;
		newHeight = ceil(height/cos(a)+sin(a)*(width-(height*tan(a))));
		newWidth = ceil(width/cos(a)+sin(a)*(height-(width*tan(a))));
		a += PI;
	}
	else
	{
		a = (a*(PI/180.0)) - PI;
		newHeight = ceil(abs(abs(height/cos(a)))+sin(a)*(width-(abs(height*tan(a)))));
		newWidth = ceil(abs(abs(width/cos(a)))+sin(a)*(height-(abs(width*tan(a)))));
		a += PI;
	}
	int newxcenter = newWidth/2;
	int newycenter = newHeight/2;

	QImage* newImg = new QImage(newWidth, newHeight, QImage::Format_RGB32);
	// Rotate the image
	for(int x = 0; x < newWidth; x++)
	{
		for(int y = 0; y < newHeight; y++)
		{
			float u = cos(-1*a)*(x-newxcenter) - sin(-1*a)*(y-newycenter)+width/2;
			float v = sin(-1*a)*(x-newxcenter) + cos(-1*a)*(y-newycenter)+height/2;
			// May not be the best variance and radius, but seems to be good
			newImg->setPixel(x,y, Utility::GaussianSample(img, u, v, 0.6, 4));
		}
	}
	return newImg;
}

QImage** MainWindow::rotate_video(float a)
{

}
