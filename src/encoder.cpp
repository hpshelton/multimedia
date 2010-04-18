#include "encoder.h"
#include "decoder.h"

//QImage* Encoder::test(QImage* img)
//{
	/*unsigned char* original = Utility::linearArray(Utility::img_to_bytes(img), img->height()*3, img->width());
	unsigned long numBytes = img->width() * img->height() * 3;
	unsigned char* decoded = Decoder::runlength_decode(Encoder::runlength_encode(original, &numBytes), &numBytes);
	printf("Run Length:\n");
	for(int i = 0; i < img->width()*img->height()*3; i++)
	{
		if(original[i] != decoded[i])
		{
			printf("%d %d %d %d %d %d %d %d\n", original[i-7], original[i-6], original[i-5], original[i-4], original[i-3], original[i-2], original[i-1], original[i]);
			printf("%d %d %d %d %d %d %d %d\n", decoded[i-7], decoded[i-6], decoded[i-5], decoded[i-4], decoded[i-3], decoded[i-2], decoded[i-1], decoded[i]);
		}
	}
	printf("Huffman:\n");
	unsigned long numBytes2 = img->width() * img->height() * 3;
	unsigned char* huffDecoded = Decoder::huffman_decode(Encoder::huffman_encode(original, &numBytes2), img->height()*3, img->width(), &numBytes2);
	for(int i = 0; i < img->width()*img->height()*3; i++)
	{
		if(original[i] != huffDecoded[i])
		{
			printf("%d %d %d %d %d %d %d %d\n", original[i-7], original[i-6], original[i-5], original[i-4], original[i-3], original[i-2], original[i-1], original[i]);
			printf("%d %d %d %d %d %d %d %d\n", huffDecoded[i-7], huffDecoded[i-6], huffDecoded[i-5], huffDecoded[i-4], huffDecoded[i-3], huffDecoded[i-2], huffDecoded[i-1], huffDecoded[i]);
		}
	}
	printf("Huffman over Run Length:\n");
	unsigned long numBytes3 = img->width() * img->height() * 3;
	unsigned char* coded = Encoder::huffman_encode(Encoder::runlength_encode(original, &numBytes3), &numBytes3);
	unsigned char* decoded3 = Decoder::runlength_decode(Decoder::huffman_decode(coded, img->height()*3, img->width(), &numBytes3), &numBytes3);
	for(int i = 0; i < img->width()*img->height()*3; i++)
	{
		if(original[i] != decoded3[i])
		{
			printf("%d %d %d %d %d %d %d %d\n", original[i-7], original[i-6], original[i-5], original[i-4], original[i-3], original[i-2], original[i-1], original[i]);
			printf("%d %d %d %d %d %d %d %d\n", decoded3[i-7], decoded3[i-6], decoded3[i-5], decoded3[i-4], decoded3[i-3], decoded3[i-2], decoded3[i-1], decoded3[i]);
		}
	}*/

	/*for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < 10; j++)
			printf("%d %d %d ", qRed(img->pixel(i, j)), qGreen(img->pixel(i,j)), qBlue(img->pixel(i,j)));
		printf("\n");
	}
	printf("\n");

	unsigned char** image = Utility::img_to_bytes(img);
	for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < 10; j++)
			printf("%d %d %d ", image[i][j*3], image[i][j*3+1], image[i][j*3+2]);
		printf("\n");
	}
	printf("\n");

	QImage* newImage = Utility::bytes_to_img(Utility::img_to_bytes(img), img->width(), img->height());
	for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < 10; j++)
			printf("%d %d %d ", qRed(newImage->pixel(i, j)), qGreen(newImage->pixel(i,j)), qBlue(newImage->pixel(i,j)));
		printf("\n");
	}
	printf("\n");

	unsigned char* compressed = huffman_encode(image, img->width(), img->height()*3);
	unsigned char** decompressed = Decoder::huffman_decode(compressed, img->width(), img->height());
	for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < 10; j++)
			printf("%d %d %d ", decompressed[i][j*3], decompressed[i][j*3+1], decompressed[i][j*3+2]);
		printf("\n");
	}

	for(int i = 0; i < img->width(); i++)
	{
		for(int j = 0; j < img->height()*3; j++)
		{
			if(image[i][j] != decompressed[i][j])
				printf("Mismatch in decompression: %d != %d\n", image[i][j], decompressed[i][j]);
		}
	}
	return Utility::bytes_to_img(decompressed, img->width(), img->height());*/
//}

void Encoder::write_ppc(QImage* img, QString filename, bool huffman, bool arithmetic, bool runlength)
{
	int width = img->width();
	int height = img->height();
	int mode = 4*runlength + 2*huffman + arithmetic;
	unsigned long numBytes = width*height*3;
	unsigned char** image = Utility::img_to_bytes(img);
	unsigned char* byte_stream = Utility::linearArray(image, height*3, width);
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
				byte_stream[index++] = (unsigned char)255;
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
	double symbol_count = 256;
	for(int i = 0; i < 256; i++)
		probabilities[i] = 1;

	int input_symbol;
	double low = 0.0, high = 1.0;
	double output_symbol;
	unsigned long output_count = 0;
	double* output_stream = (double*) malloc(*numBytes/4 * sizeof(double));

//	double probabilities[3];
//	double symbol_count = 3;
//	for(int i = 0; i < 3; i++)
//		probabilities[i] = 1;
//
//	image[0] = 1;
//	image[1] = 2;
//	image[2] = 2;
//	image[3] = 1;
//	image[4] = 1;
//	image[5] = 1;
//	image[6] = 1;
//	image[7] = 1;
//
//	int input_symbol;
//	double low = 0.0, high = 1.0;
//	double output_symbol;
//	unsigned long output_count = 0;
//	double* output_stream = (double*) malloc(3 * sizeof(double));

	for(unsigned long index = 0; index < *numBytes; index++)
//	for(unsigned long index = 0; index < 8; index++)
	{
		// TODO - CLEANUP!
		input_symbol = (int) image[index];
	//	printf("%d => ", input_symbol);
		int i = 0;
		double subintervalLow = 0;
		while(i < input_symbol)
			subintervalLow += probabilities[i++];
		subintervalLow /= symbol_count;
		double subintervalHigh = probabilities[i]/symbol_count + subintervalLow;

		double range = high - low;
		high = low + range * subintervalHigh;
		low = low + range * subintervalLow;

	//	printf("!%.15f %.15f!", low, high);

		probabilities[input_symbol]++;
		symbol_count++;

		//if(index % 7 == 0 && index > 0)
		if((index+1) % 4 == 0 && index > 0)
		{
			output_symbol = low + ((high-low) / 2.0);
			output_stream[output_count++] = output_symbol;
	//		printf(" = %.15f\n", output_symbol);
			high = 1;
			low = 0;
		}
	}

	//printf("\n\n\n\n");
	*numBytes = output_count;
	return output_stream;
}
