#!/bin/sh
FILE_NAME=OpenholoTest
SOURCE_NAME=test.cpp
OPENHOLO_INCLUDE_DIR=/content/Openholo/Reference/include
OPENHOLO_LIB_PATH=/content/Openholo/Test
FFTW_LIB_PATH=/usr/local/lib
CUDA_LIB_PATH=/usr/local/cuda/lib64


LIB_NAME=ophgen

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENHOLO_LIB_PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FFTW_LIB_PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB_PATH

g++ -fpermissive -fopenmp -g -o $FILE_NAME $SOURCE_NAME -I$OPENHOLO_INCLUDE_DIR -l$LIB_NAME

if [ -d "/Result" ]; then
	echo "build exist"
else
	echo "make directory"
	mkdir Result
fi


./$FILE_NAME
