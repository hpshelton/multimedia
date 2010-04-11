#include "decoder.h"

QImage* Decoder::read_ppc(QString filename)
{
	FILE* input;
	if(!(input = fopen(filename.toStdString().c_str(), "r")))
	{
		std::cerr << "Failed to open " << filename.toStdString() << " for reading\n";
		return NULL;
	}
	int width, height;
	unsigned long numBytes;
	fscanf(input, "%d %d %ld ", &width, &height, &numBytes);
	unsigned char* byte_stream = (unsigned char*) malloc(numBytes * sizeof(unsigned char*));
	fread(byte_stream, sizeof(unsigned char), numBytes, input);
	fclose(input);
	return Utility::bytes_to_img(huffman_decode(byte_stream, width, height), width, height);
}

unsigned char* Decoder::runlength_decode(unsigned char* bitstream)
{

}

unsigned char* Decoder::arithmetic_decode(unsigned char* bitstream)
{

}


