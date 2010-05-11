#ifndef DECODER_H
#define DECODER_H

#include <iostream>
#include <fstream>

#include <QImage>
#include <QString>

#include "utility.h"

class Decoder
{
public:
	static QImage* read_ppc(QString filename, bool CUDA);
	static QImage** read_pvc(QString filename, int* num_frames);
	static QImage** read_cif(QString filename, int* num_frames);
	static QImage** read_qcif(QString filename, int* num_frames);

	static unsigned char* huffman_decode(unsigned char* bitstream, unsigned long* numBytes);
	static unsigned char* runlength_decode(unsigned char* bitstream, unsigned long* numBytes);
	static unsigned char* arithmetic_decode(double* bitstream, unsigned long* numBytes);

	static void decompress_image(QImage* img, int* compressed, bool CUDA);
	static QImage** decompress_video(int** diff, int frames, mvec** vecArr, float compression, int height, int width);
};

#endif // DECODER_H
