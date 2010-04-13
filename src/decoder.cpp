#include "decoder.h"

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

	fscanf(input, "%d %d %d %lu ", &mode, &width, &height, &numBytes);
	unsigned char* byte_stream = (unsigned char*) malloc(numBytes * sizeof(unsigned char*));
	fread(byte_stream, sizeof(unsigned char), numBytes, input);
	fclose(input);

	if(mode % 2 == 1)
	{
		byte_stream = huffman_decode(byte_stream, width, height);
	}

	// Some bug in handling no compression output
	return Utility::bytes_to_img(Utility::blockArray(byte_stream, height*3, width), width, height);
}

unsigned char* Decoder::runlength_decode(unsigned char* bitstream)
{

}

unsigned char* Decoder::arithmetic_decode(unsigned char* bitstream)
{

}


