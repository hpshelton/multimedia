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

#include <iostream>

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

	bool video;
	int frames;

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
	void saveFile();
};

#endif // MAINWINDOW_H
