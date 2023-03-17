#!/bin/sh
if [ -d "/build" ]; then
	echo "build exist"
else
	echo "make directory"
	mkdir build
fi

cmake -B ./build
cd build
make
make install
#cd ..
#cp -p ./bin/libopenholo.so ../Reference/lib/
# for file in ./src/*.h
# do
	# cp -p "$file" "../Reference/include/"
# done

# for file in ./src/*.cuh
# do
	# cp -p "$file" "../Reference/include/"
# done
