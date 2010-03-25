#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QCloseEvent>
#include <QToolBar>
#include <QAction>
#include <QMenuBar>
#include <QFileDialog>

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
	QAction* rotateAction;
	QAction* cropAction;

	bool video;

public:
	MainWindow(QWidget *parent = 0);
	~MainWindow();

public slots:
	void rotate();
	void crop();
	void openFile();
	void saveFile();
};

#endif // MAINWINDOW_H
