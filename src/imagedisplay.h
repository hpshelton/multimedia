#ifndef IMAGEDISPLAY_H
#define IMAGEDISPLAY_H

#include <QWidget>
#include <QScrollArea>
#include <QLabel>
#include <QImage>
#include <QHBoxLayout>
#include <QCloseEvent>
#include <QScrollBar>

class ImageDisplay : public QWidget
{
private:
	QLabel* leftImage;
	QLabel* rightImage;
	QScrollArea* leftPanel;
	QScrollArea* rightPanel;
	float leftScaleFactor;
	float rightScaleFactor;

	void init();
	void adjustScrollBar(QScrollBar* scrollbar, float factor);

public:
	ImageDisplay(QWidget* parent = 0);
	ImageDisplay(QImage* image, QWidget* parent = 0);

	void setLeftImage(QImage* image);
	void setRightImage(QImage* image);
	void setLeftAndRightImages(QImage* image);

	float getScaleFactor();
	QImage* getLeftImage();
	QImage* getRightImage();
	void closeEvent(QCloseEvent* e);
	void scaleImage(float factor);
};

#endif // IMAGEDISPLAY_H
