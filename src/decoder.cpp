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
		byte_stream = huffman_decode(byte_stream, width, height, &numBytes);
	mode /= 2;

	if(mode % 2 == 1)
		byte_stream = arithmetic_decode(byte_stream);
	mode /= 2;

	if(mode % 2 == 1)
		byte_stream = runlength_decode(byte_stream, &numBytes);

	return Utility::bytes_to_img(Utility::blockArray(byte_stream, height*3, width), width, height);
}

unsigned char* Decoder::runlength_decode(unsigned char* image, unsigned long* numBytes)
{
	unsigned char previous_symbol = image[0];
	unsigned char* byte_stream = (unsigned char*) malloc(*numBytes * 260 * sizeof(unsigned char));
	int index = 0;
	int count = 1;
	for(unsigned int i = 1; i < *numBytes; i++)
	{
		if(count == 2)
		{
			int mult = (int) image[i];
			while(mult-- > 0)
				byte_stream[index++] = previous_symbol;

			previous_symbol = NULL;
			count = 0;
		}
		else if(image[i] == previous_symbol)
		{
			count++;
			byte_stream[index++] = previous_symbol;
		}
		else
		{
			previous_symbol = image[i];
			byte_stream[index++] = previous_symbol;
			count = 1;
		}
	}
	*numBytes = index;
	return byte_stream;
}

unsigned char* Decoder::arithmetic_decode(unsigned char* bitstream)
{

}


