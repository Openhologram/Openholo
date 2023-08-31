#!/bin/sh

DIR_NAME="build"

#if [ -d "$DIR_NAME" ]; then
#	echo "build exist"
#else
#	echo "make directory"
#	mkdir $DIR_NAME
#fi

if [ "$1" = "-d" ] || [ "$1" = "-D" ]; then
	cmake -B ./$DIR_NAME/Debug -DCMAKE_BUILD_TYPE=Debug
	cd $DIR_NAME/Debug
else
	cmake -B ./$DIR_NAME/Release -DCMAKE_BUILD_TYPE=Release
	cd $DIR_NAME/Release
fi

echo "Build begin"
make
make install

