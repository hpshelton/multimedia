#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	QDesktopWidget qdw;
	int screenCenterX = qdw.width() / 2;
	int screenCenterY = qdw.height() / 2;
	this->setGeometry(screenCenterX - 500, screenCenterY - 300, 1000, 600);

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

	this->toolbar = addToolBar("Editing Actions");

	this->grayscaleAction = new QAction("Grayscale", this);
	QObject::connect(this->grayscaleAction, SIGNAL(triggered()), this, SLOT(grayscale()));
	this->toolbar->addAction(grayscaleAction);

	this->rotateAction = new QAction("Rotate", this);
	QObject::connect(this->rotateAction, SIGNAL(triggered()), this, SLOT(rotate()));
	this->toolbar->addAction(rotateAction);

	this->cropAction = new QAction("Crop", this);
	QObject::connect(this->cropAction, SIGNAL(triggered()), this, SLOT(crop()));
	this->toolbar->addAction(cropAction);

	this->scaleAction = new QAction("Scale", this);
	QObject::connect(this->scaleAction, SIGNAL(triggered()), this, SLOT(scale()));
	this->toolbar->addAction(scaleAction);

	this->brightenAction = new QAction("Brighten", this);
	QObject::connect(this->brightenAction, SIGNAL(triggered()), this, SLOT(brighten()));
	this->toolbar->addAction(brightenAction);

	this->contrastAction = new QAction("Contrast", this);
	QObject::connect(this->contrastAction, SIGNAL(triggered()), this, SLOT(contrast()));
	this->toolbar->addAction(contrastAction);

	this->saturateAction = new QAction("Saturate", this);
	QObject::connect(this->saturateAction, SIGNAL(triggered()), this, SLOT(saturate()));
	this->toolbar->addAction(saturateAction);

	this->blurAction = new QAction("Blur", this);
	QObject::connect(this->blurAction, SIGNAL(triggered()), this, SLOT(blur()));
	this->toolbar->addAction(blurAction);

	this->edgeDetectAction = new QAction("Detect Edges", this);
	QObject::connect(this->edgeDetectAction, SIGNAL(triggered()), this, SLOT(edgeDetection()));
	this->toolbar->addAction(edgeDetectAction);

	this->toolbar->setFloatable(false);
	this->toolbar->setMovable(false);
	this->toolbar->setAllowedAreas(Qt::TopToolBarArea);
	this->toolbar->setToolButtonStyle(Qt::ToolButtonTextOnly);
	this->toolbar->setFixedHeight(30);
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
	delete scaleAction;
	delete brightenAction;
	delete saturateAction;
	delete contrastAction;
	delete blurAction;
	delete edgeDetectAction;
	delete grayscaleAction;
}

void MainWindow::grayscale()
{
}

void MainWindow::rotate()
{
	bool accepted;
	QString i = QInputDialog::getText(this, "Rotate", "Rotation Angle (Degrees):", QLineEdit::Normal, "", &accepted);
	if(accepted)
	{
		float a = i.toFloat(&accepted);
		if(accepted && abs(a) <= 10000)
		{
			while(a < 0)
				a += 360;
			while(a >= 360)
				a -= 360;
			// rotate image by degree a
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Rotation Angle!             ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("|angle| must be less than 10,000.");
			error->show();
		}
	}
}

void MainWindow::crop()
{
	// Instantiate QDialog to take four corner parameters
}

void MainWindow::scale()
{
	bool accepted;
	QString i = QInputDialog::getText(this, "Scale", "Scale Factor:", QLineEdit::Normal, "", &accepted);
	if(accepted)
	{
		float factor = i.toFloat(&accepted);
		if(accepted && factor >= 0.0)
		{
			// scale image by factor
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Scale Factor!               ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Scale factor must be greater than 0.");
			error->show();
		}
	}
}

void MainWindow::brighten()
{
	bool accepted;
	QString i = QInputDialog::getText(this, "Adjust Brightness", "Brightness Factor:", QLineEdit::Normal, "", &accepted);
	if(accepted)
	{
		float factor = i.toFloat(&accepted);
		if(accepted && factor >= 0.0)
		{
			// brighten image by factor
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Brightness Factor!               ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Brightness factor must be greater than 0.");
			error->show();
		}
	}
}

void MainWindow::contrast()
{
	bool accepted;
	QString i = QInputDialog::getText(this, "Adjust Contrast", "Contrast Factor:", QLineEdit::Normal, "", &accepted);
	if(accepted)
	{
		float factor = i.toFloat(&accepted);
		if(accepted && factor >= 0.0)
		{
			// adjust contrast image by factor
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Contrast Factor!                   ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Contrast factor must be greater than 0.");
			error->show();
		}
	}
}

void MainWindow::saturate()
{
	bool accepted;
	QString i = QInputDialog::getText(this, "Adjust Saturation", "Saturation Factor:", QLineEdit::Normal, "", &accepted);
	if(accepted)
	{
		float factor = i.toFloat(&accepted);
		if(accepted && factor >= 0.0)
		{
			// saturate image by factor
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Saturation Factor!                ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Saturation factor must be greater than 0.");
			error->show();
		}
	}
}

void MainWindow::blur()
{
}

void MainWindow::edgeDetection()
{
}

void MainWindow::openFile()
{
	QString fileName = QFileDialog::getOpenFileName(this, "Select a file to open", "/", "*.pgm; *.qcif");
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
	QString fileName = QFileDialog::getSaveFileName(this, "Save the edited file", "/", (this->video) ? "*.pvc" : "*.ppc");
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
