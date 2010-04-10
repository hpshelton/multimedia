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

void Encoder::write_ppc(QImage* img, bool huffman, bool arithmetic, bool runlength)
{
	unsigned char** image = Utility::img_to_bytes(img);

	unsigned char* byte_stream = huffman_encode(image, img->width(), img->height()*3);
	byte_stream = runlength_encode(byte_stream);
	// runlength > huffman
}

unsigned char* Encoder::runlength_encode(unsigned char* image)
{
	return NULL;
}

unsigned char* Encoder::arithmetic_encode(unsigned char* image)
{
	return NULL;
}