#include "encoder.h"
#include "decoder.h"

unsigned char* Encoder::runlength_encode(unsigned char* image, unsigned long* numBytes)
{
	unsigned char previous_symbol = image[0];
	unsigned char* byte_stream = (unsigned char*) malloc(*numBytes * 2 * sizeof(unsigned char));
	int index = 0;
	int count = 1;
	byte_stream[index++] = previous_symbol;
	for(unsigned int i = 1; i < *numBytes; i++)
	{
		if(image[i] == previous_symbol)
		{
			count++;
			if(count == 257)
			{
				byte_stream[index++] = (unsigned char) 255;
				count = 0;
			}
			else if(count < 3)
			{
				byte_stream[index++] = image[i];
			}
		}
		else
		{
			if(count >= 2)
			{
				byte_stream[index++] = (unsigned char)(count-2);
			}
			byte_stream[index++] = image[i];
			previous_symbol = image[i];
			count = 1;
		}
	}

	if(count >= 2)
		byte_stream[index++] = (unsigned char)(count-2);

	*numBytes = index;
	return byte_stream;
}

int* Encoder::runlength_encode_int(int* image, unsigned long* numBytes)
{
	int previous_symbol = image[0];
	int* byte_stream = (int*) malloc(*numBytes * 2 * sizeof(int));
	int index = 0;
	int count = 1;
	byte_stream[index++] = previous_symbol;
	for(unsigned int i = 1; i < *numBytes; i++)
	{
		if(image[i] == previous_symbol)
		{
			count++;
			if(count == 257)
			{
				byte_stream[index++] = 255;
				count = 0;
			}
			else if(count < 3)
			{
				byte_stream[index++] = image[i];
			}
		}
		else
		{
			if(count >= 2)
			{
				byte_stream[index++] = count - 2;
			}
			byte_stream[index++] = image[i];
			previous_symbol = image[i];
			count = 1;
		}
	}

	if(count >= 2)
		byte_stream[index++] = count - 2;

	*numBytes = index;
	return byte_stream;
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

int* Decoder::runlength_decode_int(int* image, unsigned long* numBytes)
{
	int* byte_stream = (int*) malloc(*numBytes * 260 * sizeof(int));
	int previous_symbol = image[0];
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
