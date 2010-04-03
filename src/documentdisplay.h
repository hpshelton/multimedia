#ifndef DOCUMENTDISPLAY_H
#define DOCUMENTDISPLAY_H

#include <QWidget>
#include <QScrollArea>
#include <QLabel>
#include <QImage>
#include <QHBoxLayout>
#include <QCloseEvent>
#include <QScrollBar>

class DocumentDisplay : public QWidget
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
	DocumentDisplay(QWidget* parent = 0);
	DocumentDisplay(QImage* image, QWidget* parent = 0);

	void setLeftImage(QImage* image);
	void setRightImage(QImage* image);
	void setLeftAndRightImages(QImage* image);

	float getScaleFactor();
	void closeEvent(QCloseEvent* e);
	void scaleImage(float factor);
};

#endif // DOCUMENTDISPLAY_H
