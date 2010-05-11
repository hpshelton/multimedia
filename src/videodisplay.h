#ifndef VIDEODISPLAY_H
#define VIDEODISPLAY_H

#include <QWidget>
#include <QScrollArea>
#include <QLabel>
#include <QImage>
#include <QHBoxLayout>
#include <QCloseEvent>
#include <QScrollBar>
#include <QThread>
#include <QTimer>
#include <QFocusEvent>
#include <QMouseEvent>

class VideoThread;

class VideoDisplay : public QWidget
{
	Q_OBJECT
	friend class VideoThread;

private:
	QImage** leftVideo;
	QImage** rightVideo;
	QLabel* leftFrame;
	QLabel* rightFrame;
	QScrollArea* leftPanel;
	QScrollArea* rightPanel;
	int frameNum;
	int numFrames;
	float leftScaleFactor;
	float rightScaleFactor;
	VideoThread* videoThread;

	void init(int numFrames);
	void adjustScrollBar(QScrollBar* scrollbar, float factor);

public:
	VideoDisplay(int numFrames, QWidget* parent = 0);
	VideoDisplay(QImage** video, int numFrames, QWidget* parent = 0);

	void setLeftVideo(QImage** video, int frames, bool rescale = false);
	void setRightVideo(QImage** video, int frames, bool rescale = false);
	void setLeftAndRightVideos(QImage** video, int frames = -1);

	float getScaleFactor();
	QImage** getLeftVideo();
	QImage** getRightVideo();
	void closeEvent(QCloseEvent* e);
	void scaleVideo(float factor);
	void reset();

	void focusInEvent(QFocusEvent* e);
	void mousePressEvent(QMouseEvent* e);

public slots:
	void play();
	void pause();
	void next();
	void previous();
	void videoStart();
	void videoEnd();
	void fastForward();
	void rewind();
};

class VideoThread: public QThread
{
private:
	VideoDisplay* video;
	QTimer* timer;

public:
	VideoThread(VideoDisplay* v, bool forward)
	{
		video = v;
		timer = new QTimer(this);
		connect(timer, SIGNAL(timeout()), video, forward ? SLOT(next()) : SLOT(previous()));
	};

	~VideoThread()
	{
		delete timer;
	}

	void run(int speed)
	{
		timer->start(speed);
	}

	void quit()
	{
		this->timer->stop();
	}
};
#endif // VIDEODISPLAY_H
