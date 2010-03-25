#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	this->openAction = new QAction(QIcon(":/images/open.png"), "Open", this);
	this->openAction->setStatusTip(tr("Open File"));
	this->openAction->setShortcut(tr("Ctrl+O"));
	QObject::connect(this->openAction, SIGNAL(triggered()), this, SLOT(openFile()));

	this->saveAction = new QAction(QIcon(":/images/save.png"), "Save", this);
	this->saveAction->setStatusTip(tr("Save File"));
	this->saveAction->setShortcut(tr("Ctrl+S"));
	QObject::connect(this->saveAction, SIGNAL(triggered()), this, SLOT(saveFile()));

	this->exitAction = new QAction(tr("Exit"), this);
	this->exitAction->setStatusTip(tr("Exit E-Level"));
	this->exitAction->setShortcut(tr("Ctrl+W"));
	QObject::connect(this->exitAction, SIGNAL(triggered()), this, SLOT(close()));

	this->menubar = new QMenuBar();
	QMenu* fileMenu = menubar->addMenu(tr("&File"));
	fileMenu->addAction(openAction);
	fileMenu->addAction(saveAction);
	fileMenu->addSeparator();
	fileMenu->addAction(exitAction);
	this->setMenuBar(menubar);

	this->toolbar = new QToolBar("Editing Actions", this);
	this->toolbar->setMovable(false);

	this->rotateAction = new QAction("Rotate", this);
	QObject::connect(this->rotateAction, SIGNAL(triggered()), this, SLOT(rotate()));
	this->toolbar->addAction(rotateAction);

	this->cropAction = new QAction("Crop", this);
	QObject::connect(this->cropAction, SIGNAL(triggered()), this, SLOT(crop()));
	this->toolbar->addAction(cropAction);
}

MainWindow::~MainWindow()
{
	delete menubar;
	delete toolbar;
	delete openAction;
	delete saveAction;
	delete exitAction;
	delete rotateAction;
	delete cropAction;
}

void MainWindow::rotate()
{
}

void MainWindow::crop()
{
}

void MainWindow::openFile()
{
	QString fileName = QFileDialog::getOpenFileName(this, "Select a file to open", "/", "*.jpg; *.cif");
	if(!fileName.isEmpty())
	{
		if(fileName.endsWith(".cif"))
			this->video = true;
		else
			this->video = false;
	}
}

void MainWindow::saveFile()
{
	QString fileName = QFileDialog::getSaveFileName(this, "Save the edited file", "/", "*.jpg");
	if(!fileName.isEmpty())
	{
		if(this->video)
			// Save video
			;
		else
			// Save picture
			;
	}
}
