/**
 * H. Parker Shelton
 * Adam Feinstein
 * 520.443, Digital Multimedia Coding and Processing
 * Final Project
 * March 25, 2010
 */
#include <QApplication>

#include "mainwindow.h"
#include "defines.h"

int main(int argc, char *argv[])
{
	QCoreApplication::setOrganizationName(ORGANIZATION_NAME);
	QCoreApplication::setOrganizationDomain(ORGANIZATION_DOMAIN);
	QCoreApplication::setApplicationName(PROGRAM_NAME);

	QApplication multimedia(argc, argv);
	multimedia.setWindowIcon(QIcon(":/images/movie_reel.jpg"));
	MainWindow mainWindow;
	mainWindow.setWindowTitle("Multimedia");
	mainWindow.setWindowIconText("Multimedia");

	mainWindow.raise();
	mainWindow.showNormal();
	return multimedia.exec();
}
