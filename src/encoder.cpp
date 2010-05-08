#include "encoder.h"
#include "decoder.h"

QImage* Encoder::test(QImage* img)
{
/*
	unsigned char* original;
	unsigned long numBytes;
	unsigned char* decoded, *coded;
	double* arithEncoded;
	bool failed = false;

	printf("Run Length: ");
	original = img->bits();
	numBytes = img->byteCount();
	decoded = Decoder::runlength_decode(Encoder::runlength_encode(original, &numBytes), &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
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
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
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
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");

	printf("Huffman over Run Length: ");
	original = img->bits();
	numBytes = img->byteCount();
	coded = Encoder::huffman_encode(Encoder::runlength_encode(original, &numBytes), &numBytes);
	decoded = Decoder::huffman_decode(coded, &numBytes);
	decoded = Decoder::runlength_decode(decoded, &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
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
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
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
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Huffman over Arithmetic over Run Length: ");
	original = img->bits();
	numBytes = img->byteCount();
	arithEncoded = Encoder::arithmetic_encode(Encoder::huffman_encode(Encoder::runlength_encode(original, &numBytes), &numBytes), &numBytes);
	decoded = Decoder::runlength_decode(Decoder::huffman_decode(Decoder::arithmetic_decode(arithEncoded, &numBytes), &numBytes), &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	return new QImage(decoded, img->width(), img->height(), QImage::Format_RGB32);
*/

	bool failed = false;
	unsigned long numBytes;

	printf("Arithmetic: ");
	int* original = Encoder::compress_image(img, 100, false, &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
		if(original[i] < 0)
			printf("Negative: %d\n", original[i]);

	double* compressed = Encoder::arithmetic_encode_int(original, &numBytes);
	int* decompressed = Decoder::arithmetic_decode_int(compressed, &numBytes);
	QImage* newImg = new QImage(img->width(), img->height(), QImage::Format_RGB32);
	Decoder::decompress_image(newImg, decompressed, false);

	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(original[i] != decompressed[i])
		{
			printf("Failed: %d %lu %d %d\n", i, numBytes, original[i], decompressed[i]);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	return newImg;
}

void Encoder::write_ppc(QImage* img, QString filename, bool huffman, bool arithmetic, bool runlength, int compression, bool CUDA)
{
	int width = img->width();
	int height = img->height();
	int mode = 4*runlength + 2*huffman + arithmetic;
	double* arithmetic_stream = NULL;
	unsigned long numBytes;

	FILE* output;
	if(!(output = fopen(filename.toStdString().c_str(), "w")))
	{
		std::cerr << "Failed to open " << filename.toStdString() << " for writing\n";
		return;
	}

//	if(compression == 0)
//	{
//		unsigned long numBytes = img->byteCount();
//		unsigned char* byte_stream = img->bits();
//
//		if(runlength)
//			byte_stream = runlength_encode(byte_stream, &numBytes);
//		if(huffman)
//			byte_stream = huffman_encode(byte_stream, &numBytes);
//		if(arithmetic)
//			arithmetic_stream = arithmetic_encode(byte_stream, &numBytes);
//
//		fprintf(output, "%d %d %d %lu %d", mode, width, height, numBytes, compression);
//		if(arithmetic)
//			fwrite(arithmetic_stream, sizeof(double), numBytes, output);
//		else
//			fwrite(byte_stream, sizeof(unsigned char), numBytes, output);
//		fclose(output);
//	}
//	else
//	{
//		unsigned long numBytes;
//		int* byte_stream = Encoder::compress_image(img, compression, CUDA, &numBytes); // TODO - Change 0 <= compression <= 100 into QLevel value?
//		unsigned char* byte_stream_real = (unsigned char*) malloc(2 * numBytes);
//		for(int i = 0; i < numBytes; i++)
//		{
//			unsigned char* bytes = Utility::intToChars(byte_stream[i]);
//			byte_stream_real[2*i] = bytes[0];
//			byte_stream_real[2*i+1] = bytes[1];
//		}
//
////		if(runlength)
////			byte_stream = runlength_encode_int(byte_stream, &numBytes);
////		if(huffman)
////			byte_stream = huffman_encode(byte_stream, &numBytes);
//		if(arithmetic)
//			arithmetic_stream = arithmetic_encode_int(byte_stream, &numBytes);
//
//		fprintf(output, "%d %d %d %lu %d", mode, width, height, numBytes, compression);
//		if(arithmetic)
//			fwrite(arithmetic_stream, sizeof(double), numBytes, output);
//		else
//			fwrite(byte_stream, sizeof(int), numBytes, output);
//		fclose(output);
//	}

	int* int_stream = Encoder::compress_image(img, compression, CUDA, &numBytes); // TODO - Change 0 <= compression <= 100 into QLevel value?
	unsigned char* byte_stream = (unsigned char*) malloc(numBytes * 2 * sizeof(unsigned char));
	for(unsigned int i = 0; i < numBytes; i++)
	{
		unsigned char* bytes = Utility::intToChars(int_stream[i]);
		byte_stream[2*i] = bytes[0];
		byte_stream[2*i+1] = bytes[1];
	}
	numBytes *= 2;

	if(runlength)
		byte_stream = runlength_encode(byte_stream, &numBytes);
	if(huffman)
		byte_stream = huffman_encode(byte_stream, &numBytes);
	if(arithmetic)
		arithmetic_stream = arithmetic_encode(byte_stream, &numBytes, ARITH_BREAK);

	fprintf(output, "%d %d %d %lu %d", mode, width, height, numBytes, compression);
	if(arithmetic)
		fwrite(arithmetic_stream, sizeof(double), numBytes, output);
	else
		fwrite(byte_stream, sizeof(int), numBytes, output);
	fclose(output);
}
