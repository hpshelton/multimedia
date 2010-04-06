# -------------------------------------------------
# Project created by QtCreator 2010-03-25T14:43:42
# -------------------------------------------------
TARGET = multimedia
TEMPLATE = app
HEADERS += mainwindow.h \
    defines.h \
    utility.h \
    documentdisplay.h
SOURCES += main.cpp \
    mainwindow.cpp \
    documentdisplay.cpp \
    brighten.cpp
OBJECTS_DIR = ./obj
MOC_DIR = ./obj
RCC_DIR = ./obj
RESOURCES = multimedia.qrc
macx:ICON = ../images/icon.icns

# http://forums.nvidia.com/index.php?showtopic=29539
unix { 
    INCLUDEPATH += /home/adam/NVIDIA_GPU_Computing_SDK/C/common/inc
    INCLUDEPATH += /usr/local/cuda/include
    LIBS += -lcuda \
        -lcudart \
        -L/usr/lib64 \
        -L/usr/local/cuda/lib64
    cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = nvcc \
        -c \
        -Xcompiler \
        $$join(QMAKE_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        ${QMAKE_FILE_NAME} \
        -o \
        ${QMAKE_FILE_OUT}
    cuda.depends = nvcc \
        -M \
        -Xcompiler \
        $$join(QMAKE_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        ${QMAKE_FILE_NAME} \
        | \
        sed \
        "s,^.*: ,," \
        | \
        sed \
        "s,^ *,," \
        | \
        tr \
        -d \
        '\\\n'
}
macx { 
    INCLUDEPATH += "/Developer/GPU Computing/C/common/inc"
    INCLUDEPATH += /usr/local/cuda/include
    LIBS += -lcuda \
        -lcudart \
        -L/usr/local/cuda/lib
    cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = nvcc \
        -c \
        -Xcompiler \
        $$join(QMAKE_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        ${QMAKE_FILE_NAME} \
        -o \
        ${QMAKE_FILE_OUT}
    cuda.depends = nvcc \
        -M \
        -Xcompiler \
        $$join(QMAKE_CXXFLAGS,",") \
        $$join(INCLUDEPATH,'" -I "','-I "','"') \
        ${QMAKE_FILE_NAME} \
        | \
        sed \
        "s,^.*: ,," \
        | \
        sed \
        "s,^ *,," \
        | \
        tr \
        -d \
        '\\\n'
}
cuda.input = ./CUDA/EdgeDetect/kernel_func.cu
QMAKE_EXTRA_COMPILERS += cuda
