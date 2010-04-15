#ifndef UTILITY_H
#define UTILITY_H

#include <math.h>
#include <QImage>
#include "defines.h"

class Utility
{
public:
	/* allocate 2D unsigned char array */
	static unsigned char** allocate_uchar(int r, int c)
	{
		unsigned char** p;

		if((p = (unsigned char**) malloc(r * sizeof(unsigned char*))) == NULL) {
			printf(" Error in space allocation : allocate_uchar\n");
			exit(1);
		}
		if((p[0] = (unsigned char*) malloc(c * r * sizeof(unsigned char))) == NULL) {
			printf(" Error in space allocation : allocate_uchar\n");
			exit(1);
		}

		for (int i = 1; i < r; i++)
			p[i] = p[i-1] + c;
		return p;
	}

	/* allocate 2D int array */
	static int** allocate_int(int r, int c)
	{
		int** p;

		if((p = (int**) malloc(r * sizeof(int*))) == NULL) {
			printf(" Error in space allocation : allocate_int\n");
			exit(1);
		}
		if((p[0] = (int*) malloc(c * r * sizeof(int))) == NULL) {
			printf(" Error in space allocation : allocate_int\n");
			exit(1);
		}

		for (int i = 1; i < r; i++)
			p[i] = p[i-1] + c;
		return p;
	}

	/* Returns an array that can be accessed as [width][height*3] for R, [width][height*3+1] for G, [width][height*3+2] for B */
	static unsigned char** img_to_bytes(QImage* image)
	{
		unsigned char** bytes = allocate_uchar(image->width(), image->height()*3);
		for(int r = 0; r < image->height(); r++)
		{
			for(int c = 0; c < image->width(); c++)
			{
				QRgb pixel = image->pixel(c, r);
				bytes[c][r*3] = (unsigned char)qRed(pixel);
				bytes[c][r*3 + 1] = (unsigned char)qGreen(pixel);
				bytes[c][r*3 + 2] = (unsigned char)qBlue(pixel);
			}
		}
		return bytes;
	}

	/* Inverts previous function */
	static QImage* bytes_to_img(unsigned char** bytes, int width, int height)
	{
		QImage* img = new QImage(width, height, QImage::Format_RGB32);
		for(int r = 0; r < height; r++)
		{
			for(int c = 0; c < width; c++)
			{
				int red = (int)bytes[c][r*3];
				int green = (int)bytes[c][r*3+1];
				int blue = (int)bytes[c][r*3+2];
				img->setPixel(c, r, qRgb(red, green, blue));
			}
		}
		return img;
	}

	static unsigned char** img_to_lum(QImage* image)
	{
		unsigned char** lum = allocate_uchar(image->width(), image->height());
		for(int r=0; r < image->height(); r++){
			for(int c=0; c < image->width(); c++){
				lum[c][r] = CLAMP(0.3*qRed(image->pixel(c,r)) + 0.59*qGreen(image->pixel(c,r)) + 0.11*qBlue(image->pixel(c,r)));
			}
		}
		return lum;
	}

	static char* getBinVal(int num, long int len)
	{
		int total = 0, p = 0;
		char* binVal = (char*) malloc(len * sizeof(char) + 1);
		binVal[len] = '\0';
		for(int i = 0; i < len; i++)
		{
			p = pow(2, (len-1-i));
			if(total + p <= num)
			{
				binVal[i] = '1';
				total += (int)p;
			}
			else
				binVal[i] = '0';
		}
		return binVal;
	}

	static int BintoChar(char* byte)
	{
		return  (byte[0]-48)*128 +
				(byte[1]-48)*64 +
				(byte[2]-48)*32 +
				(byte[3]-48)*16 +
				(byte[4]-48)*8 +
				(byte[5]-48)*4 +
				(byte[6]-48)*2 +
				(byte[7]-48)*1;
	}

	static char* charToBin(unsigned char byte)
	{
		char* binval = (char*) malloc(9 * sizeof(char));
		unsigned char mask = 128;
		for(int i = 0; i < 8; i++)
		{
			binval[i] = (byte & mask) ? '1' : '0';
			mask = mask / 2;
		}
		binval[8] = '\0';
		return binval;
	}

	static int numBits(char* binVal)
	{
		int i = 0;
		char c;
		while ((c = binVal[i++]) != '\0')
			;
		return i-1;
	}

	/* Returns a liner array accessed as where matrix[r][c] = array[r*height*3 + c] */
	static unsigned char* linearArray(unsigned char** matrix, int width, int height)
	{
		unsigned char* array = (unsigned char*) malloc(width*height*sizeof(unsigned char));
		int index = 0;
		for(int r = 0; r < height; r++)
		{
			for(int c = 0; c < width; c++)
				array[index++] = matrix[r][c];
		}
		return array;
	}

	/* Inverts the previous function */
	static unsigned char** blockArray(unsigned char* array, int width, int height)
	{
		unsigned char** matrix = Utility::allocate_uchar(height, width);
		int index = 0;
		for(int r = 0; r < height; r++)
		{
			for(int c = 0; c < width; c++)
				matrix[r][c] = array[index++];
		}
		return matrix;
	}

	static QRgb GaussianSample(QImage* image, float x, float y, float variance, float radius)
	{
		int width = image->width();
		int height = image->height();
		float p = pow(variance, 2);
		float coeff = 1/(2.0 * PI * p);
		float denom = (2.0 * p);
		float weight = 0, powx = 0, powy = 0;
		int r = 0, g = 0, b = 0;

		// Estimate sampling area
		int lowx = (int)floor(x-1);
		int lowy = (int)floor(y-1);
		int highx = (int)ceil(x+1);
		int highy = (int)ceil(y+1);

		// Scan estimated area
		for(int i = lowx; i <= highx; i++)
		{
			for(int j = lowy; j <= highy; j++)
			{
				// Sample area within radius
				if(i >= 0 && j >= 0 && i < width && j < height)
				{
					powx = pow(i-x, 2);
					powy = pow(j-y, 2);
					if(sqrt(powx + powy) <= radius)
					{
						QRgb pixel = image->pixel(i,j);
						weight = coeff * exp(-1 * (powx + powy)/denom);
						r += qRed(pixel) * weight;
						r = CLAMP(r);
						g += qGreen(pixel) * weight;
						g = CLAMP(g);
						b += qBlue(pixel) * weight;
						b = CLAMP(b);
					}
				}
			}
		}
		return qRgb(r, g, b);
	}
};

#endif // UTILITY_H
