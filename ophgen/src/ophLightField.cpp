#include "ophLightField.h"
//#include <dirent.h>

#define for_i(itr, oper) for(uint i=0; i<itr; i++){ oper }

int ophLF::loadLF(const char* LF_directory, const char* ext)
{
	initializeLF();

	_finddata_t data;

	string sdir = std::string("./").append(LF_directory).append("/").append("*.").append(ext);
	intptr_t ff = _findfirst(sdir.c_str(), &data);
	if (ff != -1)
	{
		int num = 0;
		Real* val = new Real;

		while (1)
		{
			string imgfullname = std::string("./").append(LF_directory).append("/").append(data.name);
			ifstream file;
			file.open(imgfullname);

			int pix = 0;
			for (; pix < resolution_image; pix++) {
				if (*val == EOF)
					break;
				file >> *val;
				*(*(LF + num) + pix) = *val;
			}

			int out = _findnext(ff, &data);
			if (out == -1)
				break;
			num++;
		}
		_findclose(ff);

		for (int i = 20000; i < 20020; i++) {
			cout << *(*LF + i) << endl;
		}
			

		if (num_image[_X]*num_image[_Y] != num + 1) {
			cout << "num_image is not matched." << endl;
		}
		delete val;
		return 1;
	}
	else
	{
		cout << "LF load was failed." << endl;
		return -1;
	}
}

void ophLF::initializeLF() {
	//if (LF[0] != nullptr) {
	//	for_i(num_image[_X] * num_image[_Y],
	//		delete[] LF[i];);
	//}
	LF = new Real*[num_image[_X] * num_image[_Y]];

	for_i(num_image[_X] * num_image[_Y],
		LF[i] = new Real[resolution_image[_X] * resolution_image[_Y]];);
	for_i(num_image[_X] * num_image[_Y],
		oph::memsetArr<Real>(*(LF + i), 0, 0, resolution_image[_X] * resolution_image[_Y]););
}


void ophLF::convertLF2ComplexField() {
	oph::Complex<Real>* complexLF = new oph::Complex<Real>;
	oph::Complex<Real>* fftLF = new oph::Complex<Real>;

	for (int i = 1; i < resolution_image[_Y]; i++)
	{                                                                                                                                                                                                          
		for (int j = 1; j < resolution_image[_X]; j++)
		{
			//complexLF = LF   
		}
	}
}