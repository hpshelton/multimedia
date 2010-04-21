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
