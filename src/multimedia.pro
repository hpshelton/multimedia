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
	blur.cpp
OBJECTS_DIR = ./obj
MOC_DIR = ./obj
RCC_DIR = ./obj
RESOURCES = multimedia.qrc
macx:ICON = ../images/icon.icns

# Cuda extra-compiler for handling files specified in the CUSOURCES variable
CUSOURCES = kernel_func.cu
unix:INCLUDEPATH += /home/adam/NVIDIA_GPU_Computing_SDK/C/common/inc
unix:INCLUDEPATH += /usr/local/cuda/include
unix:LIBS += -lcuda \
    -lcudart \
    -L/usr/lib64 \
    -L/usr/local/cuda/lib64
unix:QMAKE_CUC = /usr/local/cuda/bin/nvcc
 { 
    cu.name = Cuda \
        ${QMAKE_FILE_IN}
    cu.input = CUSOURCES
    cu.CONFIG += no_link
    cu.variable_out = OBJECTS
    isEmpty(QMAKE_CUC) { 
        QMAKE_CUC = nvcc
    }
    isEmpty(CU_DIR):CU_DIR = .
    isEmpty(QMAKE_CPP_MOD_CU):QMAKE_CPP_MOD_CU = 
    isEmpty(QMAKE_EXT_CPP_CU):QMAKE_EXT_CPP_CU = .cu
    unix:INCLUDEPATH += /usr/local/cuda/include
    unix:LIBPATH += /usr/local/cuda/lib64
    
    # QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS
    # DebugBuild:QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS_DEBUG
    # ReleaseBuild:QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS_RELEASE
    # QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS_RTTI_ON $$QMAKE_CXXFLAGS_WARN_ON $$QMAKE_CXXFLAGS_STL_ON
    # message(QMAKE_CUFLAGS: $$QMAKE_CUFLAGS)
    QMAKE_CUEXTRAFLAGS += -Xcompiler \
        $$join(QMAKE_CUFLAGS, ",") \
        $$CUFLAGS
    QMAKE_CUEXTRAFLAGS += $(DEFINES) \
        $(INCPATH) \
        $$join(QMAKE_COMPILER_DEFINES, " -D", -D)
    QMAKE_CUEXTRAFLAGS += -c
    
    # QMAKE_CUEXTRAFLAGS += -keep
    # QMAKE_CUEXTRAFLAGS += -clean
    QMAKE_EXTRA_VARIABLES += QMAKE_CUEXTRAFLAGS
    cu.commands = $$QMAKE_CUC \
        $(EXPORT_QMAKE_CUEXTRAFLAGS) \
        -o \
        $$OBJECTS_DIR/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ} \
        ${QMAKE_FILE_NAME}$$escape_expand(\n\t)
    cu.output = $$OBJECTS_DIR/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
    silent:cu.commands = @echo \
        nvcc \
        ${QMAKE_FILE_IN} \
        && \
        $$cu.commands
    QMAKE_EXTRA_COMPILERS += cu
    build_pass|isEmpty(BUILDS):cuclean.depends = compiler_cu_clean
    else:cuclean.CONFIG += recursive
    QMAKE_EXTRA_TARGETS += cuclean
}
OTHER_FILES += kernel_func.cu \
    kernels.cu
