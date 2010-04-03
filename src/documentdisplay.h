#ifndef DOCUMENTDISPLAY_H
#define DOCUMENTDISPLAY_H

#include <QWidget>
#include <QScrollArea>
#include <QLabel>
#include <QImage>
#include <QHBoxLayout>
#include <QCloseEvent>

class DocumentDisplay : public QWidget
{
	Q_OBJECT

private:
	QLabel* leftImage;
	QLabel* rightImage;
	QScrollArea* leftPanel;
	QScrollArea* rightPanel;
	float leftScaleFactor;
	float rightScaleFactor;

	void init();

public:
	DocumentDisplay(QWidget* parent = 0);
	DocumentDisplay(QImage* image, QWidget* parent = 0);

	void setLeftImage(QImage* image);
	void setRightImage(QImage* image);
	void setLeftAndRightImages(QImage* image);

	void closeEvent(QCloseEvent* e);

signals:

public slots:

};

#endif // DOCUMENTDISPLAY_H
