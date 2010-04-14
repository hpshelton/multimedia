#include "mainwindow.h"
#include <iostream>

/* cuda method is RGBA because RGB32 stores data as 0xffRRGGBB anyway
  */
QImage* MainWindow::edge_detect()
{
	QImage* img = this->file[0];

	unsigned char* CUinput;
	unsigned char* CUoutput;
	int memSize = img->byteCount();

	cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
	cutilSafeCall(cudaMalloc((void**)&CUoutput, memSize));

	cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
	edgeDetectGPU_rgba(CUoutput, CUinput, img->height(), img->width());
	cutilSafeCall(cudaMemcpy(img->bits(), CUoutput, memSize, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaFree(CUinput));
	cutilSafeCall(cudaFree(CUoutput));

	return img;

/*	float coeff[3][3] = {{-1, -1, -1},
                             {-1,  8, -1},
                             {-1, -1, -1}};

	int i, j;
	float convSumR, convSumG, convSumB;

	QImage* img = this->file[0];

	for(int y = 0; y < img->height(); y++)
	{
		for(int x = 0; x < img->width(); x++)
		{
			convSumR=0;
			convSumG=0;
			convSumB=0;
			for(i=-1; i < 2; i++){
				for(j=-1; j < 2; j++){
					if( 0<=(x+i) && (x+i) < img->width() && 0 <= (y+j) && (y+j) < img->height() ){
						convSumR += (coeff[j+1][i+1] * qRed  (img->pixel(x+i,y+j)));
						convSumG += (coeff[j+1][i+1] * qGreen(img->pixel(x+i,y+j)));
						convSumB += (coeff[j+1][i+1] * qBlue (img->pixel(x+i,y+j)));
					}
				}
			}
			img->setPixel(x,y,qRgb(CLAMP(convSumR),CLAMP(convSumG),CLAMP(convSumB)));
		}
	}
	return img;
*/}

QImage* MainWindow::edge_detect_video()
{
	return NULL;
}
