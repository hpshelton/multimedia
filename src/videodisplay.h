#ifndef VIDEODISPLAY_H
#define VIDEODISPLAY_H

#include <QWidget>
#include <QScrollArea>
#include <QLabel>
#include <QImage>
#include <QHBoxLayout>
#include <QCloseEvent>
#include <QScrollBar>

class VideoDisplay : public QWidget
{
	Q_OBJECT

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

	int getNumFrames()
	{
		return numFrames;
	}

public slots:
	void play();
	void pause();
	void videoStart();
	void videoEnd();
};
#endif // VIDEODISPLAY_H



