#ifndef ENCODER_H
#define ENCODER_H

#include <QImage>

#include "utility.h"

class Encoder
{
public:
	static void write_pgm(QImage* image, bool huffman, bool arithmetic, bool runlength);
};

#endif // ENCODER_H
