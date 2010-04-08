#include "encoder.h"

// Not pgm - custom format!
void Encoder::write_pgm(QImage* img, bool huffman, bool arithmetic, bool runlength)
{
	unsigned char** image = Utility::img_to_bytes(img);
	// runlegnth > huffman
}
