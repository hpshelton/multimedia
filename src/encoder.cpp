#include "encoder.h"
//#include "decoder.h"

/*QImage* Encoder::test(QImage* img)
{
	for(int i = 0; i < 10; i++)
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
	return Utility::bytes_to_img(decompressed, img->width(), img->height());
}*/

void Encoder::write_ppc(QImage* img, QString filename, bool huffman, bool arithmetic, bool runlength)
{
	int width = img->width();
	int height = img->height();
	unsigned int numBytes = width*height*3;

	unsigned char** image = Utility::img_to_bytes(img);
	unsigned char* byte_stream = Utility::linearArray(image, height*3, width);
	unsigned char** newImg = Utility::blockArray(byte_stream, height*3, width);
	std::cout << (int)image[10][12] << " " << (int)byte_stream[10*height*3 + 12] << " " << (int)newImg[10][12] << std::endl;
	if(runlength)
	{
		byte_stream = runlength_encode(byte_stream, &numBytes);
	}

	if(arithmetic)
	{
		byte_stream = arithmetic_encode(byte_stream, &numBytes);
	}

	if(huffman)
	{
		byte_stream = huffman_encode(byte_stream, numBytes, &numBytes);
	}

	FILE* output;
	if(!(output = fopen(filename.toStdString().c_str(), "w")))
	{
		std::cerr << "Failed to open " << filename.toStdString() << " for writing\n";
		return;
	}
	fprintf(output, "%d %d %dl ", width, height, numBytes);
	fwrite(byte_stream, sizeof(unsigned char), numBytes, output);
	fclose(output);
}

unsigned char* Encoder::runlength_encode(unsigned char* image, unsigned int* numBytes)
{
	return NULL;
}

unsigned char* Encoder::arithmetic_encode(unsigned char* image, unsigned int* numBytes)
{
	return NULL;
}
