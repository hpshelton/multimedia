#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <QImage>

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

	/* Returns an array that can be accessed as [width][height] for R, [width][height+1] for G,
	   [width][height+2] for B, and [width][height+3] for A */
	static unsigned char** img_to_bytes(QImage* image)
	{
		unsigned char** bytes = allocate_uchar(image->width(), image->height()*4);
		for(int r = 0; r < image->height(); r += 4)
		{
			for(int c = 0; c < image->width(); c++)
			{
				bytes[c][r] = (unsigned char)qRed(image->pixel(c,r));
				bytes[c][r + 1] = (unsigned char)qGreen(image->pixel(c,r));
				bytes[c][r + 2] = (unsigned char)qBlue(image->pixel(c,r));
				bytes[c][r + 3] = (unsigned char)qAlpha(image->pixel(c,r));
			}
		}
		return bytes;
	}
};

#endif // UTILITY_H
