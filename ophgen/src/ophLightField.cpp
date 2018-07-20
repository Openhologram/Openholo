#include "ophLightField.h"
//#include <dirent.h>

#define for_i(itr, oper) for(uint i=0; i<itr; i++){ oper }

int ophLF::readLFConfig(const char* LF_config) {
	return 0;
}


int ophLF::loadLF(const char* LF_directory, const char* ext)
{
	initializeLF();

	_finddata_t data;

	string sdir = std::string("./").append(LF_directory).append("/").append("*.").append(ext);
	intptr_t ff = _findfirst(sdir.c_str(), &data);
	if (ff != -1)
	{
		int num = 0;
		uchar* rgbOut;
		ivec2 sizeOut;
		int bytesperpixel;

		while (1)
		{
			//for_i(5, cout << "before: " << *(LF + num) << endl;);

			string imgfullname = std::string("./").append(LF_directory).append("/").append(data.name);

			getImgSize(sizeOut[_X], sizeOut[_Y], bytesperpixel, imgfullname.c_str());
			rgbOut = new uchar[sizeOut[_X] * sizeOut[_Y] * 3];

			loadAsImg(imgfullname.c_str(), rgbOut);					// 여기서부터 문제!

			//for_i(20, cout << *(rgbOut + i) << endl;);
			//cin.get();
			convertToFormatGray8(rgbOut, *(LF + num), sizeOut[_X], sizeOut[_Y], bytesperpixel);

			num++;
			cout << num << endl;

			//for_i(5, cout << "after: " << *(*(LF + num) + i) << endl;);

			int out = _findnext(ff, &data);
			if (out == -1)
				break;


		}
		_findclose(ff);
		cout << "LF load was successed." << endl;
		if (num_image[_X]*num_image[_Y] != num) {
			cout << "num_image is not matched." << endl;
			cin.get();
		}
		return 1;
	}
	else
	{
		cout << "LF load was failed." << endl;
		cin.get();
		return -1;
	}
}

void ophLF::generateHologram() {
	convertLF2ComplexField();
	fresnelPropagation(context_, RSplane_complex_field, holo_gen, distanceRS2Holo);
}


void ophLF::initializeLF() {
	//if (LF[0] != nullptr) {
	//	for_i(num_image[_X] * num_image[_Y],
	//		delete[] LF[i];);
	//}
	cout << "initialize LF..." << endl;

	LF = new uchar*[num_image[_X] * num_image[_Y]];

	for_i(num_image[_X] * num_image[_Y],
		LF[i] = new uchar[resolution_image[_X] * resolution_image[_Y]];);

	for_i(num_image[_X] * num_image[_Y],
		oph::memsetArr<uchar>(*(LF + i), '0', 0, resolution_image[_X] * resolution_image[_Y]-1););

	cout << "number of the images : " << num_image[_X] * num_image[_Y] << endl;
}


void ophLF::convertLF2ComplexField() {

	uint nx = num_image[_X];
	uint ny = num_image[_Y];
	uint rx = resolution_image[_X];
	uint ry = resolution_image[_Y];

	Complex<Real>* complexLF = new Complex<Real>[rx*ry];

	Complex<Real>* FFTLF = new Complex<Real>[rx*ry];

	Real randVal;
	Complex<Real> phase;

	for (uint idxRx = 0; idxRx < rx; idxRx++) {
		for (uint idxRy = 0; idxRy < ry; idxRy++) {

			for (uint idxNx = 0; idxNx < nx; idxNx++) {
				for (uint idxNy = 0; idxNy < ny; idxNy++) {

					(*(complexLF + (idxNx + nx*idxNy)))._Val[_RE] = (Real)*(*(LF + (idxNx + nx*idxNy)) + (idxNx + nx*idxNy));
				}
			}
			
			fft2(nx*ny, complexLF, OPH_FORWARD, OPH_ESTIMATE);
			fftExecute(FFTLF);
			
			for (uint idxNx = 0; idxNx < nx; idxNx++) {
				for (uint idxNy = 0; idxNy < ny; idxNy++) {

					randVal = rand((Real)0, (Real)1, idxRx*idxRy);
					phase._Val[_IM] = randVal * 2 * M_PI;

					*(RSplane_complex_field + nx*rx*ny*idxRy + nx*rx*idxNy + nx*idxRx + idxNx) = *(FFTLF+ (idxNx + nx*idxNy)) * phase.exp();
				}
			}		

		}
	}

	delete[] complexLF, FFTLF;
}