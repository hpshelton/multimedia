#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QCloseEvent>
#include <QToolBar>
#include <QAction>
#include <QMenuBar>
#include <QFileDialog>
#include <QLabel>
#include <QDesktopWidget>
#include <QInputDialog>
#include <QMessageBox>
#include <QGridLayout>
#include <QDialogButtonBox>
#include <QRadioButton>
#include <QGroupBox>
#include <QSpinBox>
#include <QImage>

#include "utility.h"
#include "documentdisplay.h"
#include "defines.h"
#include "encoder.h"

#include "cutil_inline.h"

class MainWindow : public QMainWindow
{
	Q_OBJECT

private:
	QMenuBar* menubar;
	QToolBar* toolbar;

/* Menu actions */
	QAction* openAction;
	QAction* saveAction;
	QAction* exitAction;
	QAction* closeAction;

/* Toolbar actions */
	QAction* cropAction;
	QAction* rotateAction;
	QAction* scaleAction;
	QAction* brightenAction;
	QAction* contrastAction;
	QAction* saturateAction;
	QAction* blurAction;
	QAction* edgeDetectAction;
	QAction* grayscaleAction;
	QAction* compressAction;

	QAction* zoomInAction;
	QAction* zoomOutAction;

	bool video;
	int frames;
	QImage** file;
	bool hasChanged;

	DocumentDisplay* display;

	bool displaySavePrompt();
	void toggleActions(bool);

	QImage* brighten_image(float factor);
	QImage* brighten_video(float factor);

public:
	MainWindow(QWidget *parent = 0);
	~MainWindow();

public slots:
	void grayscale();
	void crop();
	void rotate();
	void scale();
	void brighten();
	void contrast();
	void saturate();
	void blur();
	void edgeDetection();
	void compress();
	void openFile();
	bool saveFile();
	void closeFile();
	void zoomIn();
	void zoomOut();
};

#endif // MAINWINDOW_H
