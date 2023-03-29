#!/bin/sh
FILE_NAME=OpenholoTest
SOURCE_NAME=test.cpp
INCLUDE_DIR=../Reference/include
LIBRARY_DIR=../Reference/lib
LIB_NAME=ophgen

g++ -fpermissive -fopenmp -g -o $FILE_NAME $SOURCE_NAME -I$INCLUDE_DIR -I$CUDA_HOME/include -L$LIBRARY_DIR -l$LIB_NAME

./$FILE_NAME
