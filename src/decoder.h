#ifndef DECODER_H
#define DECODER_H

#include <stdio.h>
#include <iostream>
#include "utility.h"

class Decoder
{
public:
	/* process pgm header */
	static void get_pgm_image_info(FILE* fd, int* row, int* col, int* color)
	{
		char P, Five, str[256];

		fread(&P, 1, 1, fd);
		fread(&Five, 1, 1, fd);
		rewind(fd);
		if((P == 'P') && (Five == '5')) {
			fgets (str, 256, fd);
			do { fgets (str, 256, fd);}  while (str[0] == '#');
			sscanf (str, "%d%d", col, row);
			fgets (str, 256, fd);
			*color = 1;
		}
		else if((P == 'P') && (Five == '6')) {
			fgets (str, 256, fd);
			do { fgets (str, 256, fd);}  while (str[0] == '#');
			sscanf (str, "%d%d", col, row);
			fgets (str, 256, fd);
			*color = 3;
		}
		else {
			*color = 1;
			fread(col, sizeof(int), 1, fd);
			fread(row, sizeof(int), 1, fd);
		}
	};

	/* read pgm (raw with simple header) image */
	static unsigned char** read_pgm(const char* file, int* row, int* col, int* color)
	{
		FILE* fd;
		unsigned char** image;

		if((fd = fopen(file, "r")) == NULL) {
			printf(" Error opening file [%s]\n", file);
			return NULL;
		}
		get_pgm_image_info(fd, row, col, color);
		image = Utility::allocate_uchar(*row, (*col)*(*color));
		fread(image[0], 1, (*row)*(*col)*(*color), fd);
		fclose(fd);
		return image;
	};
};

#endif // DECODER_H
