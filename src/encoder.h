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
	static void write_ppc(QImage* image, QString filename, bool huffman, bool arithmetic, bool runlength);
	static void write_pvc();
	static unsigned char* huffman_encode(unsigned  char* image, unsigned long* numBytes);
	static unsigned char* runlength_encode(unsigned char* image, unsigned long* numBytes);
	static double* arithmetic_encode(unsigned char* image, unsigned long* numBytes);
};

#endif // ENCODER_H
