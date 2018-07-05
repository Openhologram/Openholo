#include "ophLightField.h"
#include <dirent.h>

int ophLF::loadLF(const char* LF_directory, const char* ext)
{
	_finddata_t data;
	string src_folder = "sample_orthographic_images";
	string sdir = std::string("./").append(src_folder).append("/").append("*.png");
	int ff = _findfirst(sdir.c_str(), &data);
	if (ff != -1)
	{
		int res = 0;
		while (res != -1)
		{
			string imgfullname = std::string("./").append(src_folder).append("/").append(data.name);
			res = _findnext(ff, &data);
		}
		// ¹Ì¿Ï¼º                                     
		_findclose(ff);
	}

	

	ifstream fin;
	fin.open("test.bmp", ios::in | ios::binary);
	int c, i;
	int* ima;
	while ((c = fin.get()) != EOF) {
		*(ima + i) = c;
		i++;
	}
	fin.close();


	return 0;
}

void ophLF::convertLF2ComplexField() {
	oph::Complex<real>* complexLF = new oph::Complex<real>;
	oph::Complex<real>* fftLF = new oph::Complex<real>;

	for (int i = 1; i < resolution_image[_Y]; i++)
	{                                                                                                                                                                                                          
		for (int j = 1; j < resolution_image[_X]; j++)
		{
			//complexLF = LF   
		}
	}
}