#include "documentdisplay.h"

DocumentDisplay::DocumentDisplay(QWidget *parent)
	: QWidget(parent)
{
	init();
}

DocumentDisplay::DocumentDisplay(QImage *image, QWidget* parent)
	: QWidget(parent)
{
	init();
	setLeftAndRightImages(image);
}

void DocumentDisplay::init()
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

	this->leftScaleFactor = 0;
	this->rightScaleFactor = 0;
}

void DocumentDisplay::setLeftImage(QImage *image)
{
	this->leftPanel->show();
	this->leftImage->setPixmap(QPixmap::fromImage(*image));
	this->leftImage->adjustSize();
	this->leftScaleFactor = 1;
}

void DocumentDisplay::setRightImage(QImage *image)
{
	this->rightPanel->show();
	this->rightImage->setPixmap(QPixmap::fromImage(*image));
	this->rightImage->adjustSize();
	this->rightScaleFactor = 1;
}

void DocumentDisplay::setLeftAndRightImages(QImage *image)
{
	setLeftImage(image);
	setRightImage(image);
}

void DocumentDisplay::closeEvent(QCloseEvent* e)
{
	this->rightPanel->hide();
	this->leftPanel->hide();
	e->ignore();
}

void DocumentDisplay::scaleImage(float factor)
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

float DocumentDisplay::getScaleFactor()
{
	if(this->leftPanel->hasFocus())
		return leftScaleFactor;
	else
		return rightScaleFactor;
}

void DocumentDisplay::adjustScrollBar(QScrollBar* scrollBar, float factor)
{
	scrollBar->setValue(int(factor * scrollBar->value()
							+ ((factor - 1) * scrollBar->pageStep()/2)));
}
