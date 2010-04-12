nvcc -c ./EdgeDetect/kernel_func.cu -I "/Developer/GPU Computing/C/common/inc" -I /usr/local/cuda/include
mv *.o ./../obj/CUDA/
