#ifndef UTILITY_H
#define UTILITY_H

#include <math.h>
#include <QImage>
#include "defines.h"

#include "mvec.h"

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

	// Converts 0 <= num <= 65,535
	static unsigned char* intToChars(int num)
	{
		int total = 0, p = 0, index = 0, char_count = 0, len = 16;
		unsigned char* chars = (unsigned char*) malloc(2 * sizeof(unsigned char));
		char buffer[8];
		for(int i = 0; i < len; i++)
		{
			p = pow(2, (len-1-i));
			if(total + p <= num)
			{
				buffer[index++] = '1';
				total += p;
			}
			else
				buffer[index++] = '0';
			if(index == 8)
			{
				index = 0;
				chars[char_count++] = Utility::BintoChar(buffer);
			}
		}
		return chars;
	}

	// Converts -32767 <= num <= 32767 correctly
	static unsigned char* shortToChars(short num)
	{
		unsigned char* chars = (unsigned char*) malloc(2 * sizeof(unsigned char));
		bool negative = (num < 0);
		num = abs(num);
		char buffer[16];
		for(int i = 15; i >= 0; i--)
		{
			buffer[i] = (num % 2 + 48);
			num = num >> 1;
		}

		char small_buffer[8];
		small_buffer[0] = (negative) ? '1' : '0';

		for(int i = 1; i < 8; i++)
			small_buffer[i] = buffer[i];
		chars[0] = Utility::BintoChar(small_buffer);

		for(int i = 0; i < 8; i++)
			small_buffer[i] = buffer[i+8];
		chars[1] = Utility::BintoChar(small_buffer);

		return chars;
	}

	// Returns signed short from two byte twos-complement representation
	static int charsToShort(unsigned char byte1, unsigned char byte2)
	{
		char* first = Utility::charToBin(byte1);
		char* second = Utility::charToBin(byte2);
		int mult = (first[0]-48) ? -1 : 1;
		return  mult*(
				(first[1]-48)*16384 +
				(first[2]-48)*8192 +
				(first[3]-48)*4096 +
				(first[4]-48)*2048 +
				(first[5]-48)*1024 +
				(first[6]-48)*512 +
				(first[7]-48)*256 +
				(second[0]-48)*128 +
				(second[1]-48)*64 +
				(second[2]-48)*32 +
				(second[3]-48)*16 +
				(second[4]-48)*8 +
				(second[5]-48)*4 +
				(second[6]-48)*2 +
				(second[7]-48)*1);
	}

	static int doubleCharToInt(unsigned char byte1, unsigned char byte2)
	{
		char* first = charToBin(byte1);
		char* second = charToBin(byte2);
		return  (first[0]-48)*32768 +
				(first[1]-48)*16384 +
				(first[2]-48)*8192 +
				(first[3]-48)*4096 +
				(first[4]-48)*2048 +
				(first[5]-48)*1024 +
				(first[6]-48)*512 +
				(first[7]-48)*256 +
				(second[0]-48)*128 +
				(second[1]-48)*64 +
				(second[2]-48)*32 +
				(second[3]-48)*16 +
				(second[4]-48)*8 +
				(second[5]-48)*4 +
				(second[6]-48)*2 +
				(second[7]-48)*1;
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

	static double psnr(unsigned char* A, unsigned char* B, int len)
	{
		double MSE = 0;

		for(int i=0; i < len; i++)
			MSE += (A[i]-B[i])*(A[i]-B[i]);
		MSE /=len;

		return 10 * log(255*255/MSE)/log(10);
	}

	static double psnr_video(QImage** A, QImage** B, int frames)
	{
		double MSE = 0;

		int len = A[0]->height() * A[0]->width()*4;

		for(int f=0; f < frames; f++)
			for(int i=0; i < len; i++){
				MSE += (A[f]->bits()[i]-B[f]->bits()[i])*(A[f]->bits()[i]-B[f]->bits()[i]);
			}
		MSE /=(len*frames);

		return 10 * log(255*255/MSE)/log(10);
	}

	static double pct_zeros(int** A, int frames, int len)
	{
		int zeros=0;

		for(int f=0; f < frames; f++){
			for(int i=0; i < len; i++){
				if(A[f][i] ==0){
					zeros++;
				}
			}
		}
		return 100*(zeros / (double)(len*frames));
	}
};

#endif // UTILITY_H
