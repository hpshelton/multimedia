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
	static void write_pvc(QImage** video, QString filename, int start_frame, int end_frame, int compression, bool huffman, bool runlength, bool arithmetic);

	static unsigned char* huffman_encode(unsigned char* image, unsigned long* numBytes);
	static unsigned char* runlength_encode(unsigned char* image, unsigned long* numBytes);
	static double* arithmetic_encode(unsigned char* image, unsigned long* numBytes);

	static int* compress_image(QImage* img, float compression, bool CUDA, unsigned long* numBytes);
	static QImage* compress_image_preview(QImage* img, float compression, double *psnr, bool CUDA);

	static int** compress_video(QImage** video, int start_frame, int end_frame, mvec*** vecArr, float compression);
	static QImage** compress_video_preview(QImage** video, int start_frame, int end_frame, float compression, double *psnr);
};

#endif // ENCODER_H
