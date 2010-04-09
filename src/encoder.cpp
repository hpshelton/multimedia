#include "encoder.h"

// Not pgm - custom format!
void Encoder::write_ppc(QImage* img, bool huffman, bool arithmetic, bool runlength)
{
	unsigned char** image = Utility::img_to_bytes(img);
	// runlength > huffman
}

unsigned char* Encoder::huffman_encode(unsigned char** image)
{

}
