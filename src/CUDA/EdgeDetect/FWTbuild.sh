nvcc -c kernel_func.cu -I /home/adam/NVIDIA_GPU_Computing_SDK/C/common/inc/ -I /usr/local/cuda/include/
g++-4.3 fwtTest.cpp -I /home/adam/NVIDIA_GPU_Computing_SDK/C/common/inc/ -I /usr/local/cuda/include/ -c
g++-4.3 fwtTest.o kernel_func.o -L/usr/lib64 -L/usr/local/cuda/lib64 -lcuda -lcudart -o fwtTest
