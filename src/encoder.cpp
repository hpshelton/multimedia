#include "encoder.h"
#include "decoder.h"
/*
QImage* Encoder::test(QImage* img)
{
	unsigned char* original;
	unsigned long numBytes;
	unsigned char* decoded, *coded;
	double* arithEncoded;
	bool failed = false;

	printf("Run Length: ");
	original = img->bits();
	numBytes = img->byteCount();
	decoded = Decoder::runlength_decode(Encoder::runlength_encode(original, &numBytes), &numBytes);
	for(int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed!\n");
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Huffman: ");
	original = img->bits();
	numBytes = img->byteCount();
	decoded = Decoder::huffman_decode(Encoder::huffman_encode(original, &numBytes), &numBytes);
	for(int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed!\n");
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Arithmetic: ");
	original = img->bits();
	numBytes = img->byteCount();
	decoded = Decoder::arithmetic_decode(Encoder::arithmetic_encode(original, &numBytes), &numBytes);
	for(int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed!\n");
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");

	printf("Huffman over Run Length: ");
	original = img->bits();
	numBytes = img->byteCount();
	coded = Encoder::huffman_encode(Encoder::runlength_encode(original, &numBytes), &numBytes);
	decoded = Decoder::runlength_decode(Decoder::huffman_decode(coded, &numBytes), &numBytes);
	for(int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed!\n");
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Arithmetic over Run Length: ");
	original = img->bits();
	numBytes = img->byteCount();
	arithEncoded = Encoder::arithmetic_encode(Encoder::runlength_encode(original, &numBytes), &numBytes);
	decoded = Decoder::runlength_decode(Decoder::arithmetic_decode(arithEncoded, &numBytes), &numBytes);
	for(int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed!\n");
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Huffman over Arithmetic: ");
	original = img->bits();
	numBytes = img->byteCount();
	arithEncoded = Encoder::arithmetic_encode(Encoder::huffman_encode(original, &numBytes), &numBytes);
	decoded = Decoder::huffman_decode(Decoder::arithmetic_decode(arithEncoded, &numBytes), &numBytes);
	for(int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed!\n");
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Huffman over Arithmetic over Runlength: ");
	original = img->bits();
	numBytes = img->byteCount();
	arithEncoded = Encoder::arithmetic_encode(Encoder::huffman_encode(Encoder::runlength_encode(original, &numBytes), &numBytes), &numBytes);
	decoded = Decoder::runlength_decode(Decoder::huffman_decode(Decoder::arithmetic_decode(arithEncoded, &numBytes), &numBytes), &numBytes);
	for(int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed!\n");
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	return new QImage(decoded, img->width(), img->height(), QImage::Format_RGB32);
}
*/
void Encoder::write_ppc(QImage* img, QString filename, bool huffman, bool arithmetic, bool runlength)
{
	int width = img->width();
	int height = img->height();
	int mode = 4*runlength + 2*huffman + arithmetic;
	unsigned long numBytes = img->byteCount();
	unsigned char* byte_stream = img->bits();
	double* arithmetic_stream = NULL;

	if(runlength)
		byte_stream = runlength_encode(byte_stream, &numBytes);
	if(huffman)
		byte_stream = huffman_encode(byte_stream, &numBytes);
	if(arithmetic)
		arithmetic_stream = arithmetic_encode(byte_stream, &numBytes);

	FILE* output;
	if(!(output = fopen(filename.toStdString().c_str(), "w")))
	{
		std::cerr << "Failed to open " << filename.toStdString() << " for writing\n";
		return;
	}
	fprintf(output, "%d %d %d %lu ", mode, width, height, numBytes);
	if(!arithmetic)
		fwrite(byte_stream, sizeof(unsigned char), numBytes, output);
	else
		fwrite(arithmetic_stream, sizeof(double), numBytes, output);
	fclose(output);
}

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

double* Encoder::arithmetic_encode(unsigned char* image, unsigned long* numBytes)
{
	// Compute initial uniform probabilities
	double probabilities[256];
	unsigned long symbol_count = 256;
	for(int i = 0; i < 256; i++)
		probabilities[i] = 1;

	int input_symbol;
	double low = 0.0, high = 1.0;
	double output_symbol;
	unsigned long output_count = 0;
	double* output_stream = (double*) malloc((*numBytes+1)/4 * sizeof(double));

	for(unsigned long index = 0; index < *numBytes; index++)
	{
		// TODO - CLEANUP!
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
