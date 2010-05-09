#include "encoder.h"
#include "decoder.h"

QImage* Encoder::test(QImage* img)
{
	unsigned long numBytes, numBytes2;
	int* int_stream = Encoder::compress_image(img, 100, false, &numBytes2);
	numBytes = numBytes2;
	unsigned char* byte_stream = (unsigned char*) malloc(numBytes * 2 * sizeof(unsigned char));
	for(unsigned int i = 0; i < numBytes; i++)
	{
//		unsigned char* bytes = Utility::intToChars(int_stream[i]);
		unsigned char* bytes = Utility::shortToChars(int_stream[i]);
		byte_stream[2*i] = bytes[0];
		byte_stream[2*i+1] = bytes[1];
	}
	numBytes *= 2;

	bool failed = false;

	printf("Run Length: ");
	numBytes = numBytes2;
	unsigned char* decoded = Decoder::runlength_decode(Encoder::runlength_encode(byte_stream, &numBytes), &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(byte_stream[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Huffman: ");
	numBytes = numBytes2;
	decoded = Decoder::huffman_decode(Encoder::huffman_encode(byte_stream, &numBytes), &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(byte_stream[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Arithmetic: ");
	numBytes = numBytes2;
	decoded = Decoder::arithmetic_decode(Encoder::arithmetic_encode(byte_stream, &numBytes), &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(byte_stream[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");

	printf("Huffman over Run Length: ");
	numBytes = numBytes2;
	unsigned char* coded = Encoder::huffman_encode(Encoder::runlength_encode(byte_stream, &numBytes), &numBytes);
	decoded = Decoder::runlength_decode(Decoder::huffman_decode(coded, &numBytes), &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(byte_stream[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Arithmetic over Run Length: ");
	numBytes = numBytes2;
	double* arithEncoded = Encoder::arithmetic_encode(Encoder::runlength_encode(byte_stream, &numBytes), &numBytes);
	decoded = Decoder::runlength_decode(Decoder::arithmetic_decode(arithEncoded, &numBytes), &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(byte_stream[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Huffman over Arithmetic: ");
	numBytes = numBytes2;
	arithEncoded = Encoder::arithmetic_encode(Encoder::huffman_encode(byte_stream, &numBytes), &numBytes);
	decoded = Decoder::huffman_decode(Decoder::arithmetic_decode(arithEncoded, &numBytes), &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(byte_stream[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	printf("Huffman over Arithmetic over Run Length: ");
	numBytes = numBytes2;
	arithEncoded = Encoder::arithmetic_encode(Encoder::huffman_encode(Encoder::runlength_encode(byte_stream, &numBytes), &numBytes), &numBytes);
	decoded = Decoder::runlength_decode(Decoder::huffman_decode(Decoder::arithmetic_decode(arithEncoded, &numBytes), &numBytes), &numBytes);
	for(unsigned int i = 0; i < numBytes; i++)
	{
		if(byte_stream[i] != decoded[i])
		{
			printf("Failed: %d %lu\n", i, numBytes);
			failed = true;
		}
	}
	if(!failed)
		printf("Passed!\n");
	failed = false;

	return img;
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

	int* int_stream = Encoder::compress_image(img, compression, CUDA, &numBytes);
	unsigned char* byte_stream = (unsigned char*) malloc(numBytes * 2 * sizeof(unsigned char));
	for(unsigned int i = 0; i < numBytes; i++)
	{
//		unsigned char* bytes = Utility::intToChars(int_stream[i]);
		unsigned char* bytes = Utility::shortToChars(int_stream[i]);
		byte_stream[2*i] = bytes[0];
		byte_stream[2*i+1] = bytes[1];
	}
	numBytes *= 2;

	if(runlength)
		byte_stream = runlength_encode(byte_stream, &numBytes);
	if(huffman)
		byte_stream = huffman_encode(byte_stream, &numBytes);
	if(arithmetic)
		arithmetic_stream = arithmetic_encode(byte_stream, &numBytes);

	fprintf(output, "%d %d %d %lu %d ", mode, width, height, numBytes, compression);
	if(arithmetic)
		fwrite(arithmetic_stream, sizeof(double), numBytes, output);
	else
		fwrite(byte_stream, sizeof(unsigned char), numBytes, output);
	fclose(output);

	free(arithmetic_stream);
	free(int_stream);
	free(byte_stream);
}

void Encoder::write_pvc(QImage** video, QString filename, int start_frame, int end_frame, int compression)
{
	int width = video[0]->width();
	int height = video[0]->height();
	int numFrames = end_frame - start_frame + 1;

	FILE* output;
	if(!(output = fopen(filename.toStdString().c_str(), "w")))
	{
		std::cerr << "Failed to open " << filename.toStdString() << " for writing\n";
		return;
	}

	mvec** motionVectors;
	int** residuals  = Encoder::compress_video(video, start_frame, end_frame, &motionVectors, compression); // Probably needs to be char

	// Linearize motion vectors for write
	short* linearMotionVectors = (short*) malloc(numFrames*CEIL(width/8.0)*CEIL(height/8.0)*sizeof(short)*2); // Assumes < 33,000
	int index = 0;
	for(int f = 0; f < numFrames; f++)
	{
		for(int i = 0; i < CEIL(width/8.0) * CEIL(height/8.0); i++)
		{
			mvec v = motionVectors[f][i];
			linearMotionVectors[index++] = (short)v.x;
			linearMotionVectors[index++] = (short)v.y;
		}
	}

	fprintf(output, "%d %d %d %d ", width, height, numFrames, compression);
	fwrite(linearMotionVectors, sizeof(short), numFrames*CEIL(width/8.0)*CEIL(height/8.0)*sizeof(short)*2, output);
	for(int f = 0; f < numFrames; f++)
		fwrite(residuals[f], sizeof(int), width*height*4, output);
	fclose(output);

	free(linearMotionVectors);
	for(int i = 0; i < numFrames; i++)
	{
		free(motionVectors[i]);
		free(residuals[i]);
	}
	free(motionVectors);
	free(residuals);
}
