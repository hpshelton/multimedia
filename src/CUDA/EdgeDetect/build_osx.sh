/usr/local/cuda/bin/nvcc -c kernel_func.cu -I "/Developer/GPU Computing/C/common/inc" -I /usr/local/cuda/include -o ../../obj/CUDA/kernel_func.cu
g++ EdgeDetect.cpp -m32 -I "/Developer/GPU Computing/C/common/inc" -I /usr/local/cuda/include/ -c -o ../../obj/CUDA/edgeDetect.o
