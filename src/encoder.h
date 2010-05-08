#ifndef ENCODER_H
#define ENCODER_H

#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <QImage>
#include <QString>

#include "utility.h"

class Encoder
{
public:
	static QImage* test(QImage*);
	static void write_ppc(QImage* image, QString filename, bool huffman, bool arithmetic, bool runlength, int compression, bool CUDA);
	static void write_pvc();
	static unsigned char* huffman_encode(unsigned  char* image, unsigned long* numBytes);
	static unsigned char* runlength_encode(unsigned char* image, unsigned long* numBytes);
	static int* runlength_encode_int(int* image, unsigned long* numBytes);
	static double* arithmetic_encode(unsigned char* image, unsigned long* numBytes);

	static int* compress_image(QImage* img, float factor, bool CUDA);
	static QImage* compress_image_preview(QImage* img, float factor, double *psnr, bool CUDA);

	static int** compress_video(QImage** video, int frames, int* vecArr, int Qlevel);
	static QImage** compress_video_preview(QImage** video, int frames, int Qlevel, double *psnr);
};

#endif // ENCODER_H
