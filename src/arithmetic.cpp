#include "encoder.h"
#include "decoder.h"

double* Encoder::arithmetic_encode(unsigned char* image, unsigned long* numBytes)
{
	double output_symbol;
	unsigned long output_count = 0;
	double* output_stream = (double*) malloc((*numBytes+1)/4 * sizeof(double));
	int input_symbol;
	double low = 0.0, high = 1.0;

	unsigned long symbol_count = 256;
	double probabilities[256];

	for(unsigned long index = 0; index < *numBytes; index++)
	{
		if(index % ARITH_BREAK == 0)
		{
			// Compute initial uniform probabilities
			symbol_count = 256;
			for(int i = 0; i < 256; i++)
				probabilities[i] = 1.0;
		}

		input_symbol = (int) image[index];
		int i = 0;
		double subintervalLow = 0;
		while(i < input_symbol)
			subintervalLow += probabilities[i++];
		subintervalLow /= symbol_count;
		double subintervalHigh = probabilities[i]/symbol_count + subintervalLow;

		double range = high - low;
		high = low + range * subintervalHigh;
		low = low + range * subintervalLow;

		probabilities[input_symbol]++;
		symbol_count++;

		if((index+1) % 4 == 0 && index > 0)
		{
			output_symbol = low + ((high-low) / 2.0);
			output_stream[output_count++] = output_symbol;
			high = 1.0;
			low = 0.0;
		}
	}

	*numBytes = output_count;
	return output_stream;
}

unsigned char* Decoder::arithmetic_decode(double* bitstream, unsigned long* numBytes)
{
	unsigned char* output_stream = (unsigned char*) malloc(*numBytes * 5 * sizeof(unsigned char));
	unsigned long output_index = 0;
	double symbol;
	double low = 0.0, high = 1.0;

	unsigned long symbol_count = 256;
	double counts[256];
	double probabilities[256];

	for(unsigned long input_index = 0; input_index < *numBytes; input_index++)
	{
		if(output_index % ARITH_BREAK == 0)
		{
			// Reset initial uniform probabilities
			symbol_count = 256;
			for(int i = 0; i < 256; i++)
			{
				counts[i] = 1.0;
				probabilities[i] = 1.0 / symbol_count;
			}
		}

		symbol =  bitstream[input_index];
		high = 1.0;
		low = 0.0;

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

			counts[i]++;
			symbol_count++;

			for(int j = 0; j < 256; j++)
				probabilities[j] = (counts[j] / symbol_count);
		}
	}

	*numBytes = output_index;
	return output_stream;
}
