#include "decoder.h"
#include <iostream>

QImage* Decoder::read_ppc(QString filename)
{
	FILE* input;
	int width, height, mode = 0;
	unsigned long numBytes;

	if(!(input = fopen(filename.toStdString().c_str(), "r")))
	{
		std::cerr << "Failed to open " << filename.toStdString() << " for reading\n";
		return NULL;
	}

	//unsigned char one, two, three;
	fscanf(input, "%d %d %d %lu", &mode, &width, &height, &numBytes);
	//printf("%d %d %d", one, two, three);

	bool arithmetic = (mode % 2 == 1);
	mode /= 2;
	bool huffman = (mode % 2 == 1);
	mode /= 2;
	bool runlength = (mode % 2 == 1);

	unsigned char* byte_stream;
	double* double_stream;
	if(!arithmetic)
	{
		byte_stream = (unsigned char*) malloc(numBytes * sizeof(unsigned char));
	//	printf("4: %d %d\n", byte_stream[0], byte_stream[1]);
		fread(byte_stream, sizeof(unsigned char), numBytes, input);
//		printf("5: %d %d\n", byte_stream[0], byte_stream[1]);
	}
	else
	{
		double_stream = (double*) malloc(numBytes * sizeof(double));
		fread(double_stream, sizeof(double), numBytes, input);
		byte_stream = arithmetic_decode(double_stream, &numBytes);
	}
	fclose(input);

	if(huffman)
		byte_stream = huffman_decode(byte_stream, &numBytes);
	if(runlength)
		byte_stream = runlength_decode(byte_stream, &numBytes);

	return new QImage(byte_stream, width, height, QImage::Format_RGB32);
}

