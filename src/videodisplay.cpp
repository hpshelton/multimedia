#include "videodisplay.h"

VideoDisplay::VideoDisplay(int num_frames, QWidget *parent)
	: QWidget(parent)
{
	init(num_frames);
}

VideoDisplay::VideoDisplay(QImage** video, int num_frames, QWidget* parent)
	: QWidget(parent)
{
	init(num_frames);
	setLeftAndRightVideos(video, 0);
}

void VideoDisplay::init(int num_frames)
{
	this->numFrames = num_frames;

	this->leftFrame = new QLabel(this);
	this->leftFrame->setBackgroundRole(QPalette::Base);
	this->leftFrame->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	this->leftFrame->setScaledContents(true);

	this->rightFrame = new QLabel(this);
	this->rightFrame->setBackgroundRole(QPalette::Base);
	this->rightFrame->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	this->rightFrame->setScaledContents(true);
	this->rightFrame->hide();

	this->leftPanel = new QScrollArea(this);
	this->leftPanel->setBackgroundRole(QPalette::Dark);
	this->leftPanel->setWidget(this->leftFrame);
	this->leftPanel->setAlignment(Qt::AlignCenter);
	this->leftPanel->setFocusPolicy(Qt::StrongFocus);
	this->leftPanel->hide();

	this->rightPanel = new QScrollArea(this);
	this->rightPanel->setBackgroundRole(QPalette::Light);
	this->rightPanel->setWidget(this->rightFrame);
	this->rightPanel->setAlignment(Qt::AlignCenter);
	this->leftPanel->setFocusPolicy(Qt::StrongFocus);
	this->rightPanel->hide();

	QHBoxLayout* layout = new QHBoxLayout;
	layout->addWidget(this->leftPanel);
	layout->addWidget(this->rightPanel);
	this->setLayout(layout);

	this->leftScaleFactor = 1.0;
	this->rightScaleFactor = 1.0;

	videoThread = new VideoThread(this, true);
}

void VideoDisplay::setLeftVideo(QImage** video, int frame, bool rescale)
{
	if(video != NULL)
	{
		this->leftVideo = video;
		this->leftPanel->show();
		this->leftFrame->setPixmap(QPixmap::fromImage(*video[frame]));
		float scale = (rescale) ? (this->leftScaleFactor = 1.0) : this->leftScaleFactor;
		if(scale != 1.0)
		{
			this->leftScaleFactor = 1.0;
			this->leftPanel->hasFocus();
			scaleVideo(scale);
		}
		else
			this->leftFrame->adjustSize();
	}
}

void VideoDisplay::setRightVideo(QImage**video, int frame, bool rescale)
{
	if(video != NULL)
	{
		this->rightVideo = video;
		this->rightPanel->show();
		this->rightFrame->setPixmap(QPixmap::fromImage(*video[frame]));
		float scale = (rescale) ? (this->rightScaleFactor = 1.0) : this->rightScaleFactor;
		if(scale != 1.0)
		{
			this->rightScaleFactor = 1.0;
			this->rightPanel->hasFocus();
			scaleVideo(scale);
		}
		else
			this->rightFrame->adjustSize();
	}
}

void VideoDisplay::setLeftAndRightVideos(QImage** video, int frame)
{
	if(video != NULL)
	{
		this->frameNum = frame;
		setLeftVideo(video, frameNum);
		setRightVideo(video, frameNum);
	}
}

void VideoDisplay::closeEvent(QCloseEvent* e)
{
	this->rightPanel->hide();
	this->leftPanel->hide();
	e->ignore();
}

void VideoDisplay::scaleVideo(float factor)
{
	if(this->leftPanel->hasFocus())
	{
		this->leftScaleFactor *= factor;
		this->leftFrame->resize(this->leftScaleFactor * this->leftFrame->pixmap()->size());
		adjustScrollBar(this->leftPanel->horizontalScrollBar(), factor);
		adjustScrollBar(this->leftPanel->verticalScrollBar(), factor);
	}
	else
	{
		this->rightScaleFactor *= factor;
		this->rightFrame->resize(this->rightScaleFactor * this->rightFrame->pixmap()->size());
		adjustScrollBar(this->rightPanel->horizontalScrollBar(), factor);
		adjustScrollBar(this->rightPanel->verticalScrollBar(), factor);
	}
}

float VideoDisplay::getScaleFactor()
{
	if(this->leftPanel->hasFocus())
		return leftScaleFactor;
	else
		return rightScaleFactor;
}

void VideoDisplay::adjustScrollBar(QScrollBar* scrollBar, float factor)
{
	scrollBar->setValue(int(factor * scrollBar->value()
							+ ((factor - 1) * scrollBar->pageStep()/2)));
}

QImage** VideoDisplay::getLeftVideo()
{
	return this->leftVideo;
}

QImage** VideoDisplay::getRightVideo()
{
	return this->rightVideo;
}

void VideoDisplay::videoStart()
{
	this->videoThread->quit();
	this->frameNum = 0;
	this->setLeftAndRightVideos(this->leftVideo, this->frameNum);
}

void VideoDisplay::videoEnd()
{
	this->videoThread->quit();
	this->frameNum = this->numFrames - 1;
	this->setLeftAndRightVideos(this->leftVideo, this->frameNum);
}

void VideoDisplay::play()
{
	this->videoThread->quit();
	this->videoThread = new VideoThread(this, true);
	this->videoThread->run(50); // ms per frame
}

void VideoDisplay::fastForward()
{
	this->videoThread->quit();
	this->videoThread = new VideoThread(this, true);
	this->videoThread->run(25);
}

void VideoDisplay::rewind()
{
	this->videoThread->quit();
	this->videoThread = new VideoThread(this, false);
	this->videoThread->run(25);
}

void VideoDisplay::pause()
{
	this->videoThread->quit();
}

void VideoDisplay::next()
{
	if(this->frameNum <  numFrames-1)
	{
		this->frameNum++;
		this->setLeftAndRightVideos(this->leftVideo, this->frameNum);
	}
	else
		this->videoThread->quit();
}

void VideoDisplay::previous()
{
	if(this->frameNum > 0)
	{
		this->frameNum--;
		this->setLeftAndRightVideos(this->leftVideo, this->frameNum);
	}
	else
		this->videoThread->quit();
}
