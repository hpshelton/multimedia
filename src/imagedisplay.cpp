#include "imagedisplay.h"

ImageDisplay::ImageDisplay(QWidget *parent)
	: QWidget(parent)
{
	init();
}

ImageDisplay::ImageDisplay(QImage *image, QWidget* parent)
	: QWidget(parent)
{
	init();
	setLeftAndRightImages(image);
}

void ImageDisplay::init()
{
	this->leftImage = new QLabel(this);
	this->leftImage->setBackgroundRole(QPalette::Base);
	this->leftImage->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	this->leftImage->setScaledContents(true);

	this->rightImage = new QLabel(this);
	this->rightImage->setBackgroundRole(QPalette::Base);
	this->rightImage->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	this->rightImage->setScaledContents(true);
	this->rightImage->hide();

	this->leftPanel = new QScrollArea(this);
	this->leftPanel->setBackgroundRole(QPalette::Dark);
	this->leftPanel->setWidget(this->leftImage);
	this->leftPanel->setAlignment(Qt::AlignCenter);
	this->leftPanel->setFocusPolicy(Qt::StrongFocus);
	this->leftPanel->hide();

	this->rightPanel = new QScrollArea(this);
	this->rightPanel->setBackgroundRole(QPalette::Dark);
	this->rightPanel->setWidget(this->rightImage);
	this->rightPanel->setAlignment(Qt::AlignCenter);
	this->leftPanel->setFocusPolicy(Qt::StrongFocus);
	this->rightPanel->hide();

	QHBoxLayout* layout = new QHBoxLayout;
	layout->addWidget(this->leftPanel);
	layout->addWidget(this->rightPanel);
	this->setLayout(layout);

	this->leftScaleFactor = 1.0;
	this->rightScaleFactor = 1.0;
}

void ImageDisplay::setLeftImage(QImage* image)
{
	if(image != NULL)
	{
		this->leftPanel->show();
		this->leftImage->setPixmap(QPixmap::fromImage(*image));
		float scale = this->leftScaleFactor;
		if(scale != 1.0)
		{
			this->leftScaleFactor = 1.0;
			this->leftPanel->hasFocus();
			scaleImage(scale);
		}
		else
			this->leftImage->adjustSize();
	}
}

void ImageDisplay::setRightImage(QImage* image)
{
	if(image != NULL)
	{
		this->rightPanel->show();
		this->rightImage->setPixmap(QPixmap::fromImage(*image));
		float scale = this->rightScaleFactor;
		if(scale != 1.0)
		{
			this->rightScaleFactor = 1;
			this->rightPanel->hasFocus();
			scaleImage(scale);
		}
		else
			this->rightImage->adjustSize();
	}
}

void ImageDisplay::setLeftAndRightImages(QImage *image)
{
	if(image != NULL)
	{
		setLeftImage(image);
		setRightImage(image);
	}
}

void ImageDisplay::closeEvent(QCloseEvent* e)
{
	this->rightPanel->hide();
	this->leftPanel->hide();
	e->ignore();
}

void ImageDisplay::scaleImage(float factor)
{
	if(this->leftPanel->hasFocus())
	{
		this->leftScaleFactor *= factor;
		leftImage->resize(this->leftScaleFactor * this->leftImage->pixmap()->size());
		adjustScrollBar(this->leftPanel->horizontalScrollBar(), factor);
		adjustScrollBar(this->leftPanel->verticalScrollBar(), factor);
	}
	else
	{
		this->rightScaleFactor *= factor;
		this->rightImage->resize(this->rightScaleFactor * this->rightImage->pixmap()->size());
		adjustScrollBar(this->rightPanel->horizontalScrollBar(), factor);
		adjustScrollBar(this->rightPanel->verticalScrollBar(), factor);
	}
}

float ImageDisplay::getScaleFactor()
{
	if(this->leftPanel->hasFocus())
		return leftScaleFactor;
	else
		return rightScaleFactor;
}

void ImageDisplay::adjustScrollBar(QScrollBar* scrollBar, float factor)
{
	scrollBar->setValue(int(factor * scrollBar->value()
							+ ((factor - 1) * scrollBar->pageStep()/2)));
}

QImage* ImageDisplay::getLeftImage()
{
	return new QImage(this->leftImage->pixmap()->toImage());
}

QImage* ImageDisplay::getRightImage()
{
	return new QImage(this->rightImage->pixmap()->toImage());
}
