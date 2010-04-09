#ifndef ENCODER_H
#define ENCODER_H

#include <QImage>

#include "utility.h"

class Encoder
{
public:
	static void write_ppc(QImage* image, bool huffman, bool arithmetic, bool runlength);
	static unsigned char* huffman_encode(unsigned char** image);

private:
	typedef struct tree_node {
		float prob;
		int sym;
		char* code;
		struct tree_node *right, *left;
	};
};

#endif // ENCODER_H
