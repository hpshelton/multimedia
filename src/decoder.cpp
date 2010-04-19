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

	fscanf(input, "%d %d %d %lu ", &mode, &width, &height, &numBytes);

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
		fread(byte_stream, sizeof(unsigned char), numBytes, input);
	}
	else
	{
		double_stream = (double*) malloc(numBytes * sizeof(double));
		fread(double_stream, sizeof(double), numBytes, input);
		byte_stream = arithmetic_decode(double_stream, &numBytes);
	}
	fclose(input);

	if(huffman)
		byte_stream = huffman_decode(byte_stream, width, height, &numBytes);
	if(runlength)
		byte_stream = runlength_decode(byte_stream, &numBytes);

	return Utility::bytes_to_img(Utility::blockArray(byte_stream, height*3, width), width, height);
}

unsigned char* Decoder::runlength_decode(unsigned char* image, unsigned long* numBytes)
{
	unsigned char* byte_stream = (unsigned char*) malloc(*numBytes * 260 * sizeof(unsigned char));
	unsigned char previous_symbol = image[0];
	byte_stream[0] = previous_symbol;
	int index = 1;
	int count = 1;
	for(unsigned int i = 1; i < *numBytes; i++)
	{
		if(count == 2)
		{
			int mult = (int) image[i];
			while(mult > 0)
			{
				mult--;
				byte_stream[index++] = previous_symbol;
			}
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

unsigned char* Decoder::arithmetic_decode(double* bitstream, unsigned long* numBytes)
{
	unsigned char* output_stream = (unsigned char*) malloc(*numBytes * 4 * sizeof(unsigned char));
	unsigned long output_index = 0;
	double symbol;
	double low = 0.0, high = 1.0;

	// Compute initial uniform probabilities
	double counts[256];
	unsigned long symbol_count = 256;
	double probabilities[256];
	for(int i = 0; i < 256; i++)
	{
		counts[i] = 1.0;
		probabilities[i] = 1.0 / symbol_count;
	}

//		double counts[3];
//		double symbol_count = 3;
//		double probabilities[3];
//		for(int i = 0; i < 3; i++)
//		{
//			counts[i] = 1;
//			probabilities[i] = counts[i] / symbol_count;
//		}

	for(unsigned long input_index = 0; input_index < *numBytes; input_index++)
//	for(unsigned long input_index = 0; input_index < 2; input_index++)
	{
		symbol =  bitstream[input_index];
		high = 1.0;
		low = 0.0;

		// TODO - CLEANUP!
		for(int j = 0; j < 4; j++)
		{
			double range = high - low;
			int i = 0;
			double subintervalHigh = 0;
			double modSYmbol = (symbol - low) / range;
			while(subintervalHigh < modSYmbol)
				subintervalHigh += probabilities[i++];
			double subintervalLow = subintervalHigh - probabilities[--i];

			unsigned char output_symbol = (unsigned char) i;
			output_stream[output_index++] = output_symbol;

			high = low + range * subintervalHigh;
			low = low + range * subintervalLow;

	//		printf("%d %0.15f %0.15f %.15f\n", output_symbol, low, high, symbol);

			counts[i]++;
			symbol_count++;
			for(int j = 0; j < 256; j++)
		//	for(int j = 0; j < 3; j++)
				probabilities[j] = (counts[j] / symbol_count);
		}
	}

	*numBytes = output_index;
//	printf("%lu\n", *numBytes);
	return output_stream;
}


