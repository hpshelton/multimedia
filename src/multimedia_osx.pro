# -------------------------------------------------
# Project created by QtCreator 2010-03-25T14:43:42
# -------------------------------------------------
TARGET = multimedia
TEMPLATE = app
HEADERS += mainwindow.h \
    defines.h \
    utility.h \
    documentdisplay.h \
    encoder.h \
    decoder.h
SOURCES += main.cpp \
    mainwindow.cpp \
    documentdisplay.cpp \
    brighten.cpp \
    encoder.cpp \
    edgeDetect.cpp \
    huffman.cpp \
    decoder.cpp \
    grayscale.cpp \
    contrast.cpp \
    saturation.cpp \
    blur.cpp \
    crop.cpp \
    scale.cpp \
    rotate.cpp \
    runlength.cpp \
    arithmetic.cpp
OBJECTS += obj/CUDA/*.o
INCLUDEPATH += "/Developer/GPU Computing/C/common/inc" \
    /usr/local/cuda/include
LIBS += -lcuda \
    -lcudart \
    -L/usr/local/cuda/lib
OBJECTS_DIR = ./obj
MOC_DIR = ./obj
RCC_DIR = ./obj
RESOURCES = multimedia.qrc
macx:ICON = ../images/icon.icns
