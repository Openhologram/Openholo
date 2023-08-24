ROOT_PATH=$(pwd)

##############################################
# get global variable
##############################################
source CMakeCache.txt
echo "CUDA_HOME: ${CUDA_HOME}"
echo "FFTW3_LIB_DIR: ${FFTW3_LIB_DIR}"

##############################################
# install fftw library
##############################################
function install_fftw_library(){
	cd $ROOT_PATH
	FFTW_LIBRARY="fftw-3.3.10"

	if [ -f $FFTW_LIBRARY.tar.gz ]; then
		echo "exist $FFTW_LIBRARY.tar.gz"
	else
		wget https://www.fftw.org/$FFTW_LIBRARY.tar.gz
	fi
	tar xzvf $FFTW_LIBRARY.tar.gz
	cd $ROOT_PATH/$FFTW_LIBRARY
	./configure --enable-threads --enable-shared --prefix=/usr/local/lib
	cmake .
	sed -i 's/ENABLE_THREADS:BOOL=OFF/ENABLE_THREADS:BOOL=ON/' CMakeCache.txt
	make
	make install
}
##############################################
# Compile openholo library
##############################################
function compile_openholo_library(){
	cd $ROOT_PATH
	mkdir -p build
 	if [ "$1" = "-d" ] || [ "$1" == "-D" ]; then
 		echo "Debug Build"
 		cmake -B ./build -DCMAKE_BUILD_TYPE=Debug
 	else
 		echo "Release Build"
 		cmake -B ./build -DCMAKE_BUILD_TYPE=Release
 	fi
	make
 	make install
}

##############################################
# execute_openholo_test
##############################################
function execute_openholo_test(){
	cd $ROOT_PATH/Colab
  	echo $ROOT_PATH/Colab

	FILE_NAME=OpenholoGeneration
	SOURCE_NAME=OpenholoGeneration.cpp
	OPENHOLO_INCLUDE_DIR=$ROOT_PATH/Reference/include
	OPENHOLO_LIB_PATH=$ROOT_PATH/bin
	FFTW_LIB_PATH=/usr/local/lib
	CUDA_LIB_PATH=/usr/local/cuda/lib64
    CUDA_INCLUDE_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/include/
	CUDA_INCLUDE_PATH_COLAB=/usr/local/cuda-11.8/targets/x86_64-linux/include/
	LIB_NAME=ophgen

	g++ -fpermissive -fopenmp -g -o $FILE_NAME $SOURCE_NAME -I$OPENHOLO_INCLUDE_DIR -I$CUDA_INCLUDE_PATH -I$CUDA_INCLUDE_PATH_COLAB -l$LIB_NAME -L$OPENHOLO_LIB_PATH

	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENHOLO_LIB_PATH
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FFTW_LIB_PATH
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB_PATH
	mkdir -p Result

	python OpenholoGeneration.py
}

if [ "$1" = "-auto" ]; then
	install_fftw_library
	compile_openholo_library
else
	echo
	echo "Select Command Type...."
	PS3="Input Number = "

	select GEN_TYPE in \
		ReBuild \
		Execute
do
 	 case $GEN_TYPE in
		ReBuild)
		echo " Openholo library rebuild "
		compile_openholo_library
		break;;

		Execute)
		echo " example execute "
		execute_openholo_test
		exit 1;;
  	esac
	done
fi
