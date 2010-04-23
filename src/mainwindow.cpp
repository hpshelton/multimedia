#include "mainwindow.h"

MainWindow::MainWindow(bool c, QWidget *parent)
	: QMainWindow(parent), frames(0)
{
	this->CUDA_CAPABLE = c;
	this->CUDA_ENABLED = c;

	QDesktopWidget qdw;
	int screenCenterX = qdw.width() / 2;
	int screenCenterY = qdw.height() / 2;
	this->setGeometry(screenCenterX - 500, screenCenterY - 350, 1000, 600);

	this->display = new DocumentDisplay(this);
	this->setCentralWidget(this->display);

	this->openAction = new QAction(QIcon(":/images/open.png"), "Open", this);
	this->openAction->setStatusTip(tr("Open File"));
	this->openAction->setShortcut(tr("Ctrl+O"));
	QObject::connect(this->openAction, SIGNAL(triggered()), this, SLOT(openFile()));

	this->closeAction = new QAction("Close", this);
	this->closeAction->setStatusTip(tr("Close File"));
	QObject::connect(this->closeAction, SIGNAL(triggered()), this, SLOT(closeFile()));

	this->saveAction = new QAction(QIcon(":/images/save.png"), "Save", this);
	this->saveAction->setStatusTip(tr("Save File"));
	this->saveAction->setShortcut(tr("Ctrl+S"));
	QObject::connect(this->saveAction, SIGNAL(triggered()), this, SLOT(saveFile()));

	this->exitAction = new QAction(tr("Exit"), this);
	this->exitAction->setStatusTip(tr("Exit E-Level"));
	this->exitAction->setShortcut(tr("Ctrl+W"));
	QObject::connect(this->exitAction, SIGNAL(triggered()), this, SLOT(close()));

	this->showPreferencesAction = new QAction(tr("Preferences"), this);
	this->showPreferencesAction->setStatusTip(tr("Show Preferences"));
	QObject::connect(this->showPreferencesAction, SIGNAL(triggered()), this, SLOT(showPreferences()));

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

	this->compressAction = new QAction("Compress", this);
	QObject::connect(this->compressAction, SIGNAL(triggered()), this, SLOT(compress()));
	this->toolbar->addAction(compressAction);

	this->toolbar->addSeparator();
	this->toolbar->setStyleSheet("QToolBar::separator{ width: 25px; }");
	this->zoomInAction = new QAction("Zoom In", this);
	this->zoomInAction->setShortcut(tr("Ctrl+="));
	QObject::connect(this->zoomInAction, SIGNAL(triggered()), this, SLOT(zoomIn()));
	this->toolbar->addAction(zoomInAction);

	this->zoomOutAction = new QAction("Zoom Out", this);
	this->zoomOutAction->setShortcut(tr("Ctrl+-"));
	QObject::connect(this->zoomOutAction, SIGNAL(triggered()), this, SLOT(zoomOut()));
	this->toolbar->addAction(zoomOutAction);

	this->toolbar->setFloatable(false);
	this->toolbar->setMovable(false);
	this->toolbar->setAllowedAreas(Qt::TopToolBarArea);
	this->toolbar->setToolButtonStyle(Qt::ToolButtonTextOnly);
	this->toolbar->setFixedHeight(30);

	this->menubar = new QMenuBar();
	QMenu* fileMenu = menubar->addMenu(tr("&File"));
	fileMenu->addAction(openAction);
	fileMenu->addAction(saveAction);
	fileMenu->addSeparator();
	fileMenu->addAction(showPreferencesAction);
	fileMenu->addAction(closeAction);
	fileMenu->addAction(exitAction);
	QMenu* viewMenu = menubar->addMenu(tr("&View"));
	viewMenu->addAction(zoomInAction);
	viewMenu->addAction(zoomOutAction);
	this->setMenuBar(menubar);

	toggleActions(false);
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
	hasChanged = true;
	if(this->video)
		this->display->setRightImage(grayscale_video());
	else
		this->display->setRightImage(grayscale_image());
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
			hasChanged = true;
			if(this->video)
				this->display->setRightImage(rotate_video(a));
			else
				this->display->setRightImage(rotate_image(a));
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Rotation Angle!             ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("|angle| must be less than 10,000.");
			error->exec();
			delete error;
		}
	}
}

void MainWindow::crop()
{
	QDialog* dialog = new QDialog(this);
	QGridLayout* layout = new QGridLayout();
	dialog->setWindowTitle("Crop");
	dialog->setModal(true);
	dialog->setFixedHeight(200);
	dialog->setMinimumWidth(375);
	dialog->setLayout(layout);

	QLineEdit* x_left = new QLineEdit(dialog);
	QLineEdit* x_right = new QLineEdit(dialog);
	QLineEdit* y_top = new QLineEdit(dialog);
	QLineEdit* y_bottom = new QLineEdit(dialog);

	layout->addWidget(new QLabel("Left: ", dialog), 1, 1, 1, 1, Qt::AlignLeft);
	layout->addWidget(new QLabel("Right: ", dialog), 1, 2, 1, 1, Qt::AlignLeft);
	layout->addWidget(new QLabel("Top: ", dialog), 3, 1, 1, 1, Qt::AlignLeft);
	layout->addWidget(new QLabel("Bottom: ", dialog), 3, 2, 1, 1, Qt::AlignLeft);

	layout->addWidget(x_left, 2, 1);
	layout->addWidget(x_right, 2, 2);
	layout->addWidget(y_top, 4, 1);
	layout->addWidget(y_bottom, 4, 2);

	QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(buttonBox, SIGNAL(accepted()), dialog, SLOT(accept()));
	connect(buttonBox, SIGNAL(rejected()), dialog, SLOT(reject()));
	layout->addWidget(buttonBox, 5, 2, 1,1, Qt::AlignRight);

	if(dialog->exec())
	{
		bool accepted1, accepted2, accepted3, accepted4;
		int x1 = x_left->text().toInt(&accepted1);
		int x2 = x_right->text().toInt(&accepted2);
		int y1 = y_top->text().toInt(&accepted3);
		int y2 = y_bottom->text().toInt(&accepted4);
		if(accepted1 && accepted2 && accepted3 && accepted4 && x1 >= 0 && x2 >= 0 && y1 >= 0 && y2 >= 0)
		{
			// crop using x1, x2, y1, y2
			hasChanged = true;
			if(this->video)
				this->display->setRightImage(crop_video(x1, x2, y1, y2));
			else
				this->display->setRightImage(crop_image(x1, x2, y1, y2));
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Crop Dimensions!               ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Crop dimensions must be non-negative.");
			error->exec();
			delete error;
		}
	}
	delete layout;
	delete buttonBox;
	delete dialog;
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
			hasChanged = true;
			if(this->video)
				this->display->setRightImage(scale_video(factor));
			else
				this->display->setRightImage(scale_image(factor));
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Scale Factor!               ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Scale factor must be greater than 0.");
			error->exec();
			delete error;
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
			hasChanged = true;
			if(this->video)
				this->display->setRightImage(brighten_video(factor));
			else
				this->display->setRightImage(brighten_image(factor));
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Brightness Factor!               ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Brightness factor must be greater than 0.");
			error->exec();
			delete error;
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
			hasChanged = true;
			if(this->video)
				this->display->setRightImage(contrast_video(factor));
			else
				this->display->setRightImage(contrast_image(factor));
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Contrast Factor!                   ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Contrast factor must be greater than 0.");
			error->exec();
			delete error;
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
			hasChanged = true;
			if(this->video)
				this->display->setRightImage(saturate_video(factor));
			else
				this->display->setRightImage(saturate_image(factor));
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Saturation Factor!                ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Saturation factor must be greater than 0.");
			error->exec();
			delete error;
		}
	}
}

void MainWindow::blur()
{
	hasChanged = true;
	if(this->video)
	   this->display->setRightImage(blur_video());
	else
		this->display->setRightImage(blur_image());
}

void MainWindow::edgeDetection()
{
	hasChanged = true;
	if(this->video)
	   this->display->setRightImage(edge_detect_video());
		else
	{
		this->display->setRightImage(edge_detect());
	}
}

void MainWindow::compress()
{
	bool accepted;
	QString i = QInputDialog::getText(this, "Compress", "Compression Ratio (Percentage):", QLineEdit::Normal, "", &accepted);
	if(accepted)
	{
		float factor = i.toFloat(&accepted);
		if(accepted && factor >= 0.0 && factor <= 100)
		{
					if(this->video)
						this->display->setRightImage(compress_video(factor));
					else
						this->display->setRightImage(compress_image(factor));
					hasChanged = true;
		}
		else
		{
			QMessageBox* error = new QMessageBox(this);
			error->setText("Invalid Compression Ratio!                           ");
			error->setIcon(QMessageBox::Critical);
			error->setInformativeText("Compression ratio must be greater than 0 and less than 100.");
			error->exec();
			delete error;
		}
	}
}

void MainWindow::openFile()
{
	closeFile();
	QString fileName = QFileDialog::getOpenFileName(this, "Select a file to open", "/", "*.pgm *.qcif *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.ppc");
	if(!fileName.isEmpty())
	{
		if(fileName.endsWith(".qcif"))
		{
			this->video = true;
		}
		else
		{
			this->video = false;
			this->frames = 1;
			this->file = (QImage**) malloc(sizeof(QImage*));
			if(fileName.endsWith(".ppc"))
			{
				this->file[0] = Decoder::read_ppc(fileName);
				if(this->file[0] != NULL)
					this->file[0] = new QImage(this->file[0]->convertToFormat(QImage::Format_RGB32));
			}
			else
			{
				QImage img(fileName);
				this->file[0] = new QImage(img.convertToFormat(QImage::Format_RGB32));
			}
		}
		if(this->file[0] != NULL)
		{
			this->saveAction->setEnabled(true);
			this->closeAction->setEnabled(true);
			this->display->setLeftAndRightImages(this->file[0]);
			this->hasChanged = false;
			toggleActions(true);
			Encoder::test(this->file[0]);
		}
	}
}

bool MainWindow::saveFile()
{
	QString fileName = QFileDialog::getSaveFileName(this, "Save the edited file", "/", (this->video) ? "*.pvc" : "*.ppc");
	if(!fileName.isEmpty())
	{
		QDialog* dialog = new QDialog(this);
		QGridLayout* layout = new QGridLayout();
		dialog->setWindowTitle("Save Options");
		dialog->setModal(true);
		dialog->setLayout(layout);

		QGroupBox* groupBox = new QGroupBox("Compression Options");
		QVBoxLayout *vbox = new QVBoxLayout();
		groupBox->setLayout(vbox);

		QCheckBox* radio1 = new QCheckBox("Huffman Coding");
		QCheckBox* radio2 = new QCheckBox("Run-Length Coding");
		QCheckBox* radio3 = new QCheckBox("Arithmetic Coding");

		QSpinBox* compressionBox = new QSpinBox(dialog);
		compressionBox->setMinimum(0);
		compressionBox->setMaximum(100);
		compressionBox->setValue(0);
		vbox->addWidget(radio1);
		vbox->addWidget(radio2);
		vbox->addWidget(radio3);
		vbox->addWidget(new QLabel("Compression Ratio (Percentage): ", dialog));
		vbox->addWidget(compressionBox);
		vbox->addStretch(1);

		if(fileName.endsWith(".jpg") || fileName.endsWith(".jpeg") || fileName.endsWith(".bmp")
			|| fileName.endsWith(".tif") || fileName.endsWith(".tiff" ))
		{
			radio1->setEnabled(false);
			radio2->setEnabled(false);
			radio3->setEnabled(false);
		}

		QGroupBox* groupBoxRight = new QGroupBox("Frame Options");
		QVBoxLayout *vboxRight = new QVBoxLayout();
		groupBoxRight->setLayout(vboxRight);

		QSpinBox* frame_start = new QSpinBox(dialog);
		frame_start->setMinimum(0);
		frame_start->setMaximum(100); // INSERT ACTUAL VIDEO LEGNTH
		QSpinBox* frame_end = new QSpinBox(dialog);
		frame_end->setMinimum(0);
		frame_end->setMaximum(100); // INSERT ACTUAL VIDEO LEGNTH
		vboxRight->addWidget(new QLabel("Start: ", dialog));
		vboxRight->addWidget(frame_start);
		vboxRight->addWidget(new QLabel("End: ", dialog));
		vboxRight->addWidget(frame_end);

		QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
		connect(buttonBox, SIGNAL(accepted()), dialog, SLOT(accept()));
		connect(buttonBox, SIGNAL(rejected()), dialog, SLOT(reject()));

		layout->addWidget(groupBox, 1, 1, 1, 1);
		layout->addWidget(groupBoxRight, 1, 2, 1, 1);
		layout->addWidget(buttonBox, 2, 2, 1, 1, Qt::AlignRight);

		if(!this->video)
			groupBoxRight->hide();

		if(dialog->exec())
		{
			bool huffman = radio1->isChecked();
			bool arithmetic = radio3->isChecked();
			bool runlength = radio2->isChecked();
			int compression = compressionBox->text().toInt();

			if(this->video)
			{
				// Save video
				int start_frame = frame_start->text().toInt();
				int end_frame = frame_end->text().toInt();
				if(end_frame < start_frame || start_frame < 0 || end_frame > this->frames)
				{
					QMessageBox* error = new QMessageBox(this);
					error->setText("Invalid Frame Settings!                         ");
					error->setIcon(QMessageBox::Critical);
					error->setInformativeText("End frame must be greater than start frame.");
					error->exec();
					delete error;
				}
				if(compression > 0)
					compress_video(compression);
				// write video
			}
			else
			{
				if(fileName.endsWith(".jpg") || fileName.endsWith(".jpeg") || fileName.endsWith(".bmp")
					|| fileName.endsWith(".tif") || fileName.endsWith(".tiff" ))
				{
					QImage* img = this->display->getRightImage();
					img->save(fileName, 0, 100-compression);
					hasChanged = false;
				}
				else
				{				
					// Save picture in ppc format
					if(compression > 0)
						Encoder::write_ppc(compress_image(compression), fileName, huffman, arithmetic, runlength);
					else
						Encoder::write_ppc(this->display->getRightImage(), fileName, huffman, arithmetic, runlength);
					if(!fileName.endsWith(".ppc"))
						fileName += ".ppc";
					hasChanged = false;
				}
			}
		}
		delete radio1;
		delete radio2;
		delete radio3;
		delete frame_start;
		delete frame_end;
		delete vbox;
		delete vboxRight;
		delete buttonBox;
		delete groupBox;
		delete groupBoxRight;
		delete layout;
		delete dialog;
	}
	return !hasChanged;
}

void MainWindow::closeFile()
{
	if(!this->hasChanged || displaySavePrompt())
	{
		this->saveAction->setEnabled(false);
		this->closeAction->setEnabled(false);
		toggleActions(false);
		hasChanged = false;
		this->display->close();
	}
}

bool MainWindow::displaySavePrompt()
{
	QMessageBox* error = new QMessageBox(this);
	error->setText("The current file has not been saved!                         ");
	error->setIcon(QMessageBox::Critical);
	error->setInformativeText("Would you like to save your changes?");
	error->setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
	int ret = error->exec();
	delete error;
	if(ret == QMessageBox::Save)
		return saveFile();
	else if(ret == QMessageBox::Discard)
		return true;
	else
		return false;
}

void MainWindow::closeEvent(QCloseEvent* e)
{
	if(!this->hasChanged || displaySavePrompt())
		e->accept();
	else
		e->ignore();
}

void MainWindow::toggleActions(bool b)
{
	this->saveAction->setEnabled(b);
	this->closeAction->setEnabled(b);
	this->cropAction->setEnabled(b);
	this->rotateAction->setEnabled(b);
	this->scaleAction->setEnabled(b);
	this->brightenAction->setEnabled(b);
	this->contrastAction->setEnabled(b);
	this->saturateAction->setEnabled(b);
	this->blurAction->setEnabled(b);
	this->edgeDetectAction->setEnabled(b);
	this->grayscaleAction->setEnabled(b);
	this->compressAction->setEnabled(b);
	this->zoomInAction->setEnabled(b);
	this->zoomOutAction->setEnabled(b);
}

void MainWindow::zoomIn()
{
	this->display->scaleImage(ZOOM_IN_FACTOR);
}

void MainWindow::zoomOut()
{
	this->display->scaleImage(ZOOM_OUT_FACTOR);
}

void MainWindow::showPreferences()
{
	QDialog* dialog = new QDialog(this);
	QGridLayout* layout = new QGridLayout();
	dialog->setWindowTitle("Preferences");
	dialog->setModal(true);
	dialog->setLayout(layout);

	QGroupBox* groupBox = new QGroupBox("CUDA-Acceleration Options");
	QVBoxLayout *vbox = new QVBoxLayout();
	groupBox->setLayout(vbox);

	QCheckBox* radio1 = new QCheckBox("Enable CUDA Accleration");
	radio1->setDisabled(!this->CUDA_CAPABLE);
	radio1->setChecked(this->CUDA_ENABLED);
	vbox->addWidget(radio1);
	QObject::connect(radio1, SIGNAL(clicked(bool)), this, SLOT(enableCUDA(bool)));
	layout->addWidget(groupBox);

	dialog->exec();

	delete radio1;
	delete vbox;
	delete layout;
	delete dialog;
}

void MainWindow::enableCUDA(bool b)
{
	this->CUDA_ENABLED = b;
}
