#include "mainwindow.h"

QImage* MainWindow::compress_image(float factor)
{
    QImage* img = this->display->getRightImage();
    int width = img->width();
    int height = img->height();

    if(CUDA_CAPABLE && CUDA_ENABLED)
    {
        ;
    }
    else
    {
        ;
    }
}

QImage* MainWindow::compress_video(float factor)
{
    QImage* img = this->display->getRightImage();
    int width = img->width();
    int height = img->height();

    if(CUDA_CAPABLE && CUDA_ENABLED)
    {
        ;
    }
    else
    {
        ;
    }
}
