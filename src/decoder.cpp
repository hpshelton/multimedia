#include "decoder.h"
#include <iostream>

QImage* Decoder::read_ppc(QString filename, bool CUDA)
{
	FILE* input;
	int width, height, mode, compression = 0;
	unsigned long numBytes;

	if(!(input = fopen(filename.toStdString().c_str(), "r")))
	{
		std::cerr << "Failed to open " << filename.toStdString() << " for reading\n";
		return NULL;
	}
	fscanf(input, "%d %d %d %lu %d", &mode, &width, &height, &numBytes, &compression);

	bool arithmetic = (mode % 2 == 1);
	mode /= 2;
	bool huffman = (mode % 2 == 1);
	mode /= 2;
	bool runlength = (mode % 2 == 1);

	printf("%d\n", compression);

	if(compression == 0)
	{
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
			byte_stream = huffman_decode(byte_stream, &numBytes);
		if(runlength)
			byte_stream = runlength_decode(byte_stream, &numBytes);

		return new QImage(byte_stream, width, height, QImage::Format_RGB32);
	}
	else
	{
		int* byte_stream = (int*) malloc(numBytes * sizeof(int));
		fread(byte_stream, sizeof(int), numBytes, input);
		QImage* img = new QImage(width, height, QImage::Format_RGB32);
		Decoder::decompress_image(img, byte_stream, CUDA);
		return img;
	}
}

QImage** Decoder::read_qcif(QString filename, int* frame_num)
{
	// Open file
	FILE* video;
	if((video = fopen(filename.toStdString().c_str(), "r")) == NULL)
	{
		std::cerr << "Error opening file [%s] for reading!\n";
		return NULL;
	}

	*frame_num = 0;
	int max_frames = 50;
	QImage** frames = (QImage**) malloc(max_frames * sizeof(QImage*));
	unsigned int Y_BLOCK_SIZE = QCIF_WIDTH * QCIF_HEIGHT;
	unsigned int U_BLOCK_SIZE = (QCIF_WIDTH * QCIF_HEIGHT)/4;
	unsigned int V_BLOCK_SIZE = (QCIF_WIDTH * QCIF_HEIGHT)/4;
	unsigned char** Y = Utility::allocate_uchar(QCIF_HEIGHT, QCIF_WIDTH);
	unsigned char** U, **V;

	// Read YUV color space blocks
	while((fread(Y[0], sizeof(unsigned char), Y_BLOCK_SIZE, video)) == Y_BLOCK_SIZE)
	{
		U = Utility::allocate_uchar(QCIF_HEIGHT/2, QCIF_WIDTH/2);
		V = Utility::allocate_uchar(QCIF_HEIGHT/2, QCIF_WIDTH/2);
		if(fread(U[0], sizeof(unsigned char), U_BLOCK_SIZE, video) != U_BLOCK_SIZE)
		{
			std::cerr << "Read error in U block, frame " << *frame_num << std::endl;
			return NULL;
		}
		if(fread(V[0], sizeof(unsigned char), V_BLOCK_SIZE, video) != V_BLOCK_SIZE)
		{
			std::cerr << "Read error in V block, frame " << *frame_num << std::endl;
			return NULL;
		}

		// Convert color space YUV to RGB
		QImage* frame = new QImage(QCIF_WIDTH, QCIF_HEIGHT, QImage::Format_RGB32);
		for(int r = 0; r < QCIF_HEIGHT; r++)
		{
			for(int c = 0; c < QCIF_WIDTH; c++)
			{
				int blue = CLAMP(1.164*(Y[r][c] - 16) + 2.018*(U[r/2][c/2]- 128));
				int green = CLAMP(1.164*(Y[r][c] - 16) - 0.813*(V[r/2][c/2] - 128) - 0.391*(U[r/2][c/2] - 128));
				int red = CLAMP(1.164*(Y[r][c] - 16) + 1.596*(V[r/2][c/2] - 128));
				frame->setPixel(c, r, qRgb(red,green,blue));
			}
		}
		// Add image to movie
		if(*frame_num == max_frames)
		{
			// Copy memory and extend array
			QImage** enlarged_frames = (QImage**) malloc(max_frames * 2 * sizeof(QImage*));
			memcpy(enlarged_frames, frames, max_frames * sizeof(QImage*));
			delete frames;
			frames = enlarged_frames;
		}
		frames[(*frame_num)++] = frame;

		delete Y;
		delete U;
		delete V;
		Y = Utility::allocate_uchar(QCIF_HEIGHT, QCIF_WIDTH);
	}
	fclose(video);
	return frames;
}

QImage** Decoder::read_cif(QString filename, int* frame_num)
{
	// Open file
	FILE* video;
	if((video = fopen(filename.toStdString().c_str(), "r")) == NULL)
	{
		std::cerr << "Error opening file [%s] for reading!\n";
		return NULL;
	}

	*frame_num = 0;
	int max_frames = 150;
	QImage** frames = (QImage**) malloc(max_frames * sizeof(QImage*));
	unsigned int Y_BLOCK_SIZE = CIF_WIDTH * CIF_HEIGHT;
	unsigned int U_BLOCK_SIZE = (CIF_WIDTH * CIF_HEIGHT)/4;
	unsigned int V_BLOCK_SIZE = (CIF_WIDTH * CIF_HEIGHT)/4;
	unsigned char** Y = Utility::allocate_uchar(CIF_HEIGHT, CIF_WIDTH);
	unsigned char** U, **V;

	// Read YUV color space blocks
	while((fread(Y[0], sizeof(unsigned char), Y_BLOCK_SIZE, video)) == Y_BLOCK_SIZE)
	{
		U = Utility::allocate_uchar(CIF_HEIGHT/2, CIF_WIDTH/2);
		V = Utility::allocate_uchar(CIF_HEIGHT/2, CIF_WIDTH/2);
		if(fread(U[0], sizeof(unsigned char), U_BLOCK_SIZE, video) != U_BLOCK_SIZE)
		{
			std::cerr << "Read error in U block, frame " << *frame_num << std::endl;
			return NULL;
		}
		if(fread(V[0], sizeof(unsigned char), V_BLOCK_SIZE, video) != V_BLOCK_SIZE)
		{
			std::cerr << "Read error in V block, frame " << *frame_num << std::endl;
			return NULL;
		}

		// Convert color space YUV to RGB
		QImage* frame = new QImage(CIF_WIDTH, CIF_HEIGHT, QImage::Format_RGB32);
		for(int r = 0; r < CIF_HEIGHT; r++)
		{
			for(int c = 0; c < CIF_WIDTH; c++)
			{
				int blue = CLAMP(1.164*(Y[r][c] - 16) + 2.018*(U[r/2][c/2]- 128));
				int green = CLAMP(1.164*(Y[r][c] - 16) - 0.813*(V[r/2][c/2] - 128) - 0.391*(U[r/2][c/2] - 128));
				int red = CLAMP(1.164*(Y[r][c] - 16) + 1.596*(V[r/2][c/2] - 128));
				frame->setPixel(c, r, qRgb(red,green,blue));
			}
		}
		// Add image to movie
		if(*frame_num == max_frames)
		{
			// Copy memory and extend array
			QImage** enlarged_frames = (QImage**) malloc(max_frames * 2 * sizeof(QImage*));
			memcpy(enlarged_frames, frames, max_frames * sizeof(QImage*));
			delete frames;
			frames = enlarged_frames;
		}
		frames[(*frame_num)++] = frame;

		delete Y;
		delete U;
		delete V;
		Y = Utility::allocate_uchar(CIF_HEIGHT, CIF_WIDTH);
	}
	fclose(video);
	return frames;
}

