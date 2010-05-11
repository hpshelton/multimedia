#ifndef IMAGEDISPLAY_H
#define IMAGEDISPLAY_H

#include <QWidget>
#include <QScrollArea>
#include <QLabel>
#include <QImage>
#include <QHBoxLayout>
#include <QCloseEvent>
#include <QScrollBar>
#include <QFocusEvent>
#include <QMouseEvent>
#include <QKeyEvent>

class ImageDisplay : public QWidget
{
private:
	QLabel* leftImage;
	QLabel* rightImage;
	QScrollArea* leftPanel;
	QScrollArea* rightPanel;
	float leftScaleFactor;
	float rightScaleFactor;

	QFrame* unselected;
	QFrame* selected;

	void init();
	void adjustScrollBar(QScrollBar* scrollbar, float factor);

public:
	ImageDisplay(QWidget* parent = 0);
	ImageDisplay(QImage* image, QWidget* parent = 0);

	void setLeftImage(QImage* image, bool rescale = false);
	void setRightImage(QImage* image, bool rescale = false);
	void setLeftAndRightImages(QImage* image);

	float getScaleFactor();
	QImage* getLeftImage();
	QImage* getRightImage();
	void closeEvent(QCloseEvent* e);
	void scaleImage(float factor);

	void focusInEvent(QFocusEvent* e);
	void mousePressEvent(QMouseEvent* e);
};

#endif // IMAGEDISPLAY_H
