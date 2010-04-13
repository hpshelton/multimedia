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
	static QImage* read_ppc(QString filename);
	static unsigned char* huffman_decode(unsigned char* bitstream, int width, int height, unsigned long* numBytes);
	static unsigned char* runlength_decode(unsigned char* bitstream, unsigned long* numBytes);
	static unsigned char* arithmetic_decode(unsigned char* bitstream);
};

#endif // DECODER_H
