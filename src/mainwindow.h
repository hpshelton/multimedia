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
#include <QTime>

#include "utility.h"
#include "imagedisplay.h"
#include "videodisplay.h"
#include "defines.h"
#include "encoder.h"
#include "decoder.h"

#include "cutil_inline.h"

extern "C" void CUquantize(float* x, int Qlevel, int maxval, int len);
extern "C" void CUzeroOut(int* x, float threshold, int len);
extern "C" void CUtranspose(float* d_odata, float* d_idata, int col, int row);
extern "C" void CUsetToVal(unsigned char* x, int len, int val);
extern "C" void CUedgeDetect(unsigned char* input, unsigned char* output, int row, int col);
extern "C" void CUblur(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUbrighten(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUgreyscale(unsigned char* output, unsigned char* input, int row, int col);
extern "C" void CUsaturate(unsigned char* output, unsigned char* input, int row, int col, float factor);
extern "C" void CUfwt97_2D_rgba(int* outputInt, unsigned char* input, int row, int col);
extern "C" void CUiwt97_2D_rgba(unsigned char* output, int* input, int row, int col);

typedef struct mvec{
	int x;
	int y;
} mvec ;

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
	QToolBar* videobar;

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

	QAction* resetAction;

	QAction* timerText;

	/* Video control actions */
	QAction* play;
	QAction* pause;
	QAction* start;
	QAction* end;
	QAction* next;
	QAction* previous;
	QAction* fastForward;
	QAction* rewind;

	bool video;
	int frames;
	int compression;
	QImage** file;
	bool hasChanged;
	QTime timer;

	ImageDisplay* image_display;
	VideoDisplay* video_display;

	bool displaySavePrompt();
	void toggleActions(bool);
	void closeEvent(QCloseEvent *);

	QImage* crop_image(QImage* img, int x1, int x2, int y1, int y2);
	QImage** crop_video(int x1, int x2, int y1, int y2);

	QImage* grayscale_image(QImage* img);
	QImage** grayscale_video();

	QImage* scale_image(QImage* img, float factor);
	QImage** scale_video(float factor);

	QImage* brighten_image(QImage* img, float factor);
	QImage** brighten_video(float factor);

	QImage* contrast_image(QImage* img, float factor);
	QImage** contrast_video(float factor);

	QImage* saturate_image(QImage* img, float factor);
	QImage** saturate_video(float factor);

	QImage* edge_detect(QImage* img);
	QImage** edge_detect_video();

	QImage* blur_image(QImage* img);
	QImage** blur_video();

	QImage* rotate_image(QImage* img, float angle);
	QImage** rotate_video(float angle);

	int* compress_image(QImage* img, float factor);
	void decompress_image(QImage* img, int* compressed);
	QImage* compress_preview(QImage* img, float factor, double *psnr);
	int** compress_video(QImage** video, mvec*** vecArr, int Qlevel);
	QImage** decompress_video(int** diff, mvec** vecArr, int Qlevel, int height, int width);
	QImage** compress_video_preview(int Qlevel, double *psnr);

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
	void reset();
};

#endif
