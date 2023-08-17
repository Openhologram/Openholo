#!/bin/sh

DIR_NAME="build"

if [ -d "$DIR_NAME" ]; then
	echo "build exist"
else
	echo "make directory"
	mkdir $DIR_NAME
fi

if [ "$1" = "-d" ] || [ "$1" = "-D" ]; then
	echo "Debug build"
	cmake -B ./$DIR_NAME -DCMAKE_BUILD_TYPE=Debug
else
	echo "Release build"
	cmake -B ./$DIR_NAME -DCMAKE_BUILD_TYPE=Release
fi

cd build

echo "Build begin"
make
make install

