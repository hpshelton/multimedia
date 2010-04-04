#include <cutil_inline.h>
#include "kernels.cu"

extern "C" void edgeDetectGPU(unsigned char* input, unsigned char* output, int row, int col);

void edgeDetectGPU(unsigned char* input, unsigned char* output, int row, int col)
{
	edge_detect<<<row,col>>>(input, output, row, col);
}
