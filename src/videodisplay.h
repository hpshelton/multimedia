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

public slots:
	void play();
	void pause();
	void next();
	void previous();
	void videoStart();
	void videoEnd();
};

class VideoThread: public QThread
{
private:
	VideoDisplay* video;
	QTimer* timer;

public:
	VideoThread(VideoDisplay* v)
	{
		video = v;
		timer = new QTimer(this);
	};

	~VideoThread()
	{
		delete timer;
	}

	void run()
	{
		connect(timer, SIGNAL(timeout()), video, SLOT(next()));
		timer->start(50);
	}

	void quit()
	{
		disconnect(timer, SIGNAL(timeout()), video, SLOT(next()));
	}
};
#endif // VIDEODISPLAY_H



