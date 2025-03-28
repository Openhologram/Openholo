ROOT_PATH=$(pwd)

##############################################
# get global variable
##############################################
function print_external_variable(){
  source ${ROOT_PATH}/Colab/build.env
  echo "CUDA_HOME: ${CUDA_HOME}"
  echo "FFTW3_LIB_DIR: ${FFTW3_LIB_DIR}"
}

##############################################
# install fftw library
##############################################
function install_fftw_library(){
#	cd $ROOT_PATH
#	FFTW_LIBRARY="fftw-3.3.10"

#	if [ -f $FFTW_LIBRARY.tar.gz ]; then
#		echo "exist $FFTW_LIBRARY.tar.gz"
#	else
#		wget https://www.fftw.org/$FFTW_LIBRARY.tar.gz
#	fi
#	tar xzvf $FFTW_LIBRARY.tar.gz
#	cd $ROOT_PATH/$FFTW_LIBRARY
#	./configure --enable-threads --enable-shared --prefix=/usr/local/lib
#	cmake .
#	sed -i 's/ENABLE_THREADS:BOOL=OFF/ENABLE_THREADS:BOOL=ON/' CMakeCache.txt
#	make
#	make install
	cp ./lib/* /usr/local/lib/ 
}
##############################################
# Compile openholo library
##############################################
function compile_openholo_library(){
  DIR_NAME="build"
	cd $ROOT_PATH
  if [ "$1" = "-d" ] || [ "$1" = "-D" ]; then
	  cmake -B ./$DIR_NAME/Debug -DCMAKE_BUILD_TYPE=Debug
	  cd $DIR_NAME/Debug
  else
	  cmake -B ./$DIR_NAME/Release -DCMAKE_BUILD_TYPE=Release
	  cd $DIR_NAME/Release
  fi
	make
 	make install
}

##############################################
# execute_openholo_test
##############################################
function execute_openholo_test(){
  source ${ROOT_PATH}/Colab/build.env
	cd $ROOT_PATH/Colab
  	echo $ROOT_PATH/Colab

	FILE_NAME=OpenholoGeneration
	SOURCE_NAME=OpenholoGeneration.cpp
	OPENHOLO_INCLUDE_DIR=$ROOT_PATH/Reference/include
	OPENHOLO_LIB_PATH=$ROOT_PATH/bin
  
  CUDA_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include/
	CUDA_INCLUDE_PATH_COLAB=$CUDA_HOME/targets/x86_64-linux/include/
	LIB_NAME=ophgen

	g++ -fpermissive -fopenmp -g -o $FILE_NAME $SOURCE_NAME -I$OPENHOLO_INCLUDE_DIR -I$CUDA_INCLUDE_PATH -I$CUDA_INCLUDE_PATH_COLAB -l$LIB_NAME -L$OPENHOLO_LIB_PATH

	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENHOLO_LIB_PATH
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FFTW3_LIB_DIR
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
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
		Build \
		Execute
do
 	 case $GEN_TYPE in
		Build)
		echo " Openholo library build "
		compile_openholo_library
    print_external_variable
		break;;

		Execute)
		echo " example execute "
		execute_openholo_test
		exit 1;;
  	esac
	done
fi

