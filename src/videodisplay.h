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
private:
	QImage** leftVideo;
	QImage** rightVideo;
	QLabel* leftFrame;
	QLabel* rightFrame;
	QScrollArea* leftPanel;
	QScrollArea* rightPanel;
	int frameNum;
	float leftScaleFactor;
	float rightScaleFactor;

	void init();
	void adjustScrollBar(QScrollBar* scrollbar, float factor);

public:
	VideoDisplay(QWidget* parent = 0);
	VideoDisplay(QImage** video, QWidget* parent = 0);

	void setLeftVideo(QImage** video, bool rescale = false);
	void setRightVideo(QImage** video, bool rescale = false);
	void setLeftAndRightVideos(QImage** video);

	float getScaleFactor();
	QImage** getLeftVideo();
	QImage** getRightVideo();
	void closeEvent(QCloseEvent* e);
	void scaleVideo(float factor);
};
#endif // VIDEODISPLAY_H



