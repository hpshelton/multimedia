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
#include <QCheckBox>
#include <QSpinBox>
#include <QImage>

#include "utility.h"
#include "documentdisplay.h"
#include "defines.h"
#include "encoder.h"
#include "decoder.h"

#include "cutil_inline.h"

extern "C" void CUquantize(float* x, int Qlevel, int maxval, int len);
extern "C" void CUzeroOut(float* x, float threshold, int len);
extern "C" void CUtranspose(float* d_odata, float* d_idata, int col, int row);
extern "C" void CUsetToVal(unsigned char* x, int len, int val);
extern "C" void CUedgeDetect(unsigned char* input, unsigned char* output, int row, int col);
extern "C" void CUblur(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUbrighten(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUgreyscale(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUsaturate(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUfwt97_2D(float* output, unsigned char* input, int row, int col);
extern "C" void CUiwt97_2D(unsigned char* output, float* input, int row, int col);

class MainWindow : public QMainWindow
{
	Q_OBJECT

private:
	/* Set automatically if the computer has a CUDA-capable GPU */
	bool CUDA_CAPABLE;
	/* Set dynamically in the Preferences pane for demo purposes; initially true */
	bool CUDA_ENABLED;

	QMenuBar* menubar;
	QToolBar* toolbar;

/* Menu actions */
	QAction* openAction;
	QAction* saveAction;
	QAction* exitAction;
	QAction* closeAction;
	QAction* showPreferencesAction;

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
	void closeEvent(QCloseEvent *);

	QImage* crop_image(int x1, int x2, int y1, int y2);
	QImage* crop_video(int x1, int x2, int y1, int y2);

	QImage* grayscale_image();
	QImage* grayscale_video();

	QImage* scale_image(float factor);
	QImage* scale_video(float factor);

	QImage* brighten_image(float factor);
	QImage* brighten_video(float factor);

	QImage* contrast_image(float factor);
	QImage* contrast_video(float factor);

	QImage* saturate_image(float factor);
	QImage* saturate_video(float factor);

	QImage* edge_detect();
	QImage* edge_detect_video();

	QImage* blur_image();
	QImage* blur_video();

	QImage* rotate_image(float angle);
	QImage* rotate_video(float angle);

        QImage* compress_image(float factor);
        QImage* compress_video(float factor);

public:
	MainWindow(bool c, QWidget *parent = 0);
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
	void openFile();
	bool saveFile();
	void closeFile();
	void zoomIn();
	void zoomOut();
	void showPreferences();
        void enableCUDA(bool b);
        void compress();
};

#endif // MAINWINDOW_H
