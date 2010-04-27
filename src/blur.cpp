#include "mainwindow.h"

QImage* MainWindow::blur_image()
{
	QImage* img = this->image_display->getRightImage();
	int width = img->width();
	int height = img->height();

	if(CUDA_CAPABLE && CUDA_ENABLED)
	{
		unsigned char* CUinput;
		unsigned char* CUoutput;
		int memSize = img->byteCount();

		cutilSafeCall(cudaMalloc((void**)&CUinput, memSize));
		cutilSafeCall(cudaMalloc((void**)&CUoutput, memSize));

		cutilSafeCall(cudaMemcpy(CUinput, img->bits(), memSize, cudaMemcpyHostToDevice));
		CUblur(CUoutput, CUinput, height, width);
		cutilSafeCall(cudaMemcpy(img->bits(), CUoutput, memSize, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaFree(CUinput));
		cutilSafeCall(cudaFree(CUoutput));

		return img;
	}
	else
	{
		QImage* newImg = new QImage(*img);

		// Weight masks
		float mask[3][3] = { {1/16.0, 2/16.0, 1/16.0}, {2/16.0, 4/16.0, 2/16.0}, {1/16.0, 2/16.0, 1/16.0} };
		float maskEdge[3][3]= { {1/12.0, 2/12.0, 1/12.0}, {2/12.0, 4/12.0, 2/12.0}, {1/12.0, 2/12.0, 1/12.0} };
		float maskCorner[3][3] = { {1/9.0, 2/9.0, 1/9.0}, {2/9.0, 4/9.0, 2/9.0}, {1/9.0, 2/9.0, 1/9.0} };

		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				int poutr = 0, poutg = 0, poutb = 0;

				if(x == 0)
				{
					if(y == 0) // top left corner
					{
						for(int i = 1; i <= 2; i++)
						{
							for(int j = 1; j <= 2; j++)
							{
								QRgb pixel = img->pixel(x+(j-1), y+(i-1));
								poutr += maskCorner[i][j]*qRed(pixel);
								poutg += maskCorner[i][j]*qGreen(pixel);
								poutb += maskCorner[i][j]*qBlue(pixel);
							}
						}
					}
					else if(y == height - 1) // bottom left corner
					{
						for(int i = 0; i <= 1; i++)
						{
							for(int j = 1; j <= 2; j++)
							{
								QRgb pixel = img->pixel(x+(j-1), y+(i-1));
								poutr += maskCorner[i][j]*qRed(pixel);
								poutg += maskCorner[i][j]*qGreen(pixel);
								poutb += maskCorner[i][j]*qBlue(pixel);
							}
						}
					}
					else // left edge
					{
						for(int i = 0; i <= 2; i++)
						{
							for(int j = 1; j <= 2; j++)
							{
								QRgb pixel = img->pixel(x+(j-1), y+(i-1));
								poutr += maskEdge[i][j]*qRed(pixel);
								poutg += maskEdge[i][j]*qGreen(pixel);
								poutb += maskEdge[i][j]*qBlue(pixel);
							}
						}
					}
				}
				else if(x == width - 1)
				{
					if(y == 0) // top right corner
					{
						for(int i = 1; i <= 2; i++)
						{
							for(int j = 0; j <= 1; j++)
							{
								QRgb pixel = img->pixel(x+(j-1), y+(i-1));
								poutr += maskCorner[i][j]*qRed(pixel);
								poutg += maskCorner[i][j]*qGreen(pixel);
								poutb += maskCorner[i][j]*qBlue(pixel);
							}
						}
					}
					else if(y == height - 1) // bottom right corner
					{
						for(int i = 0; i <= 1; i++)
						{
							for(int j = 0; j <= 1; j++)
							{
								QRgb pixel = img->pixel(x+(j-1), y+(i-1));
								poutr += maskCorner[i][j]*qRed(pixel);
								poutg += maskCorner[i][j]*qGreen(pixel);
								poutb += maskCorner[i][j]*qBlue(pixel);
							}
						}
					}
					else // right edge
					{
						for(int i = 0; i <= 2; i++)
						{
							for(int j = 0; j <= 1; j++)
							{
								QRgb pixel = img->pixel(x+(j-1), y+(i-1));
								poutr += maskEdge[i][j]*qRed(pixel);
								poutg += maskEdge[i][j]*qGreen(pixel);
								poutb += maskEdge[i][j]*qBlue(pixel);
							}
						}
					}
				}
				else // 0 < x < width
				{
					if(y == 0) // top edge
					{
						for(int i = 1; i <= 2; i++)
						{
							for(int j = 0; j <= 2; j++)
							{
								QRgb pixel = img->pixel(x+(j-1), y+(i-1));
								poutr += maskEdge[i][j]*qRed(pixel);
								poutg += maskEdge[i][j]*qGreen(pixel);
								poutb += maskEdge[i][j]*qBlue(pixel);
							}
						}
					}
					else if(y == height - 1) // bottom edge
					{
						for(int i = 0; i <= 1; i++)
						{
							for(int j = 0; j <= 2; j++)
							{
								QRgb pixel = img->pixel(x+(j-1), y+(i-1));
								poutr += maskEdge[i][j]*qRed(pixel);
								poutg += maskEdge[i][j]*qGreen(pixel);
								poutb += maskEdge[i][j]*qBlue(pixel);
							}
						}
					}
					else // middle
					{
						for(int i = 0; i <= 2; i++)
						{
							for(int j = 0; j <= 2; j++)
							{
								QRgb pixel = img->pixel(x+(j-1), y+(i-1));
								poutr += mask[i][j]*qRed(pixel);
								poutg += mask[i][j]*qGreen(pixel);
								poutb += mask[i][j]*qBlue(pixel);
							}
						}
					}
				}

				poutr = CLAMP(poutr);
				poutg = CLAMP(poutg);
				poutb = CLAMP(poutb);

				newImg->setPixel(x, y, qRgb(poutr, poutg, poutb));
			}
		}
		return newImg;
	}
}

QImage** MainWindow::blur_video()
{
	return NULL;
}

