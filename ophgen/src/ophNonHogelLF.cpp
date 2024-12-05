#include "ophNonHogelLF.h"
#include "include.h"
#include "sys.h"
#include "tinyxml2.h"

#ifdef _WIN64
#include <io.h>
#include <direct.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#endif

ophNonHogelLF::ophNonHogelLF(void)
	: num_image(ivec2(0, 0))
	, resolution_image(ivec2(0, 0))
	, distanceRS2Holo(0.0)
	, fieldLens(0.0)
	, is_ViewingWindow(false)
	, nImages(-1)
	, nBufferX(0)
	, nBufferY(0)	
	, LF(nullptr)
	, FToverUV_LF(nullptr)
	, WField(nullptr)
	, Hologram(nullptr)
	, LF_directory(nullptr)
	, ext(nullptr)
{
	LOG("*** LIGHT FIELD : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

void ophNonHogelLF::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

bool ophNonHogelLF::readConfig(const char* fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	bool bRet = true;
	auto begin = CUR_TIME;

	using namespace tinyxml2;
	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("<FAILED> Wrong file ext.\n");
		return false;
	}
	if (xml_doc.LoadFile(fname) != XML_SUCCESS)
	{
		LOG("<FAILED> Loading file.\n");
		return false;
	}

	xml_node = xml_doc.FirstChild();

	char szNodeName[32] = { 0, };
	sprintf(szNodeName, "FieldLength");
	// about viewing window
	auto next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&fieldLens))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	// about image
	sprintf(szNodeName, "Image_NumOfX");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&num_image[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Image_NumOfY");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&num_image[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Image_Width");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&resolution_image[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Image_Height");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&resolution_image[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Distance");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&distanceRS2Holo))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
	initialize();
	return bRet;
}

int ophNonHogelLF::loadLF(const char* directory, const char* exten)
{
	initializeLF();
#ifdef _WIN64
	_finddata_t data;

	string sdir = std::string(LF_directory).append("\\").append("*.").append(exten);
	intptr_t ff = _findfirst(sdir.c_str(), &data);
	if (ff != -1)
	{
		int num = 0;
		uchar* rgbOut;
		ivec2 sizeOut;
		int bytesperpixel;

		while (true)
		{
			string imgfullname = std::string(LF_directory).append("\\").append(data.name);
			getImgSize(sizeOut[_X], sizeOut[_Y], bytesperpixel, imgfullname.c_str());

			rgbOut = loadAsImg(imgfullname.c_str());

			if (rgbOut == 0) {
				LOG("<FAILED> Load image.");
				return -1;
			}

			convertToFormatGray8(rgbOut, *(LF + num), sizeOut[_X], sizeOut[_Y], bytesperpixel);
			delete[] rgbOut; // solved memory leak.
			num++;
			int out = _findnext(ff, &data);
			if (out == -1)
				break;
		}
		_findclose(ff);
#else
	string sdir;
	DIR* dir = nullptr;
	if (directory[0] != '/') {
		char buf[PATH_MAX] = { 0, };
		if (getcwd(buf, sizeof(buf)) != nullptr) {
			sdir = sdir.append(buf).append("/").append(directory);
		}
	}
	else
		sdir = string(directory);
	string ext = string(exten);

	if ((dir = opendir(sdir.c_str())) != nullptr) {

		int num = 0;
		ivec2 sizeOut;
		int bytesperpixel;
		struct dirent* ent;

		// Add file
		int cnt = 0;
		vector<string> fileList;
		while ((ent = readdir(dir)) != nullptr) {
			string filePath;
			filePath = filePath.append(sdir.c_str()).append("/").append(ent->d_name);
			if (filePath != "." && filePath != "..") {
				struct stat fileInfo;
				if (stat(filePath.c_str(), &fileInfo) == 0 && S_ISREG(fileInfo.st_mode)) {
					if (filePath.substr(filePath.find_last_of(".") + 1) == ext) {
						fileList.push_back(filePath);
						cnt++;
					}
				}
			}
		}
		closedir(dir);
		std::sort(fileList.begin(), fileList.end());

		uchar* rgbOut;
		for (size_t i = 0; i < fileList.size(); i++)
		{
			// to do
			getImgSize(sizeOut[_X], sizeOut[_Y], bytesperpixel, fileList[i].c_str());
			int size = (((sizeOut[_X] * bytesperpixel) + 3) & ~3) * sizeOut[_Y];

			rgbOut = loadAsImg(fileList[i].c_str());

			if (rgbOut == 0) {
				LOG("<FAILED> Load image.");
				return -1;
			}
			convertToFormatGray8(rgbOut, LF[i], sizeOut[_X], sizeOut[_Y], bytesperpixel);
			delete[] rgbOut; // solved memory leak.
		}

#endif
		if (num_image[_X] * num_image[_Y] != num) {
			LOG("<FAILED> Not matching image.");
		}
		return 1;
	}
	else
	{
		LOG("<FAILED> Load image.");
		return -1;
	}
}

void ophNonHogelLF::preprocessLF()
{
	resetBuffer();
	setBuffer();
	fourierTransformOverUVLF();
}

void ophNonHogelLF::generateHologram()
{
	resetBuffer();

	LOG("1) Algorithm Method : Non-hogel based hologram generation from Light Field\n");
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
	);
	LOG("3) Random Phase Use : %s\n", GetRandomPhase() ? "Y" : "N");
	LOG("4) Number of Images : %d x %d\n", num_image[_X], num_image[_Y]);

	auto begin = CUR_TIME;
	if (m_mode & MODE_GPU)
	{
		LOG("Not implement GPU version");
	}
	else
	{
		setBuffer();
		makeRandomWField();
		convertLF2ComplexFieldUsingNonHogelMethod();
		Real distance = 0.0;
		for (uint ch = 0; ch < context_.waveNum; ch++)
			Fresnel_FFT(Hologram, complex_H[ch], context_.wave_length[ch], distance);
			//fresnelPropagation(Hologram, complex_H[ch], distance, ch); //distanceRS2Holo
	}

	LOG("Total Elapsed Time: %.5lf (sec)\n", ELAPSED_TIME(begin, CUR_TIME));
}

void ophNonHogelLF::generateHologram(double thetaX, double thetaY)
{
	resetBuffer();

	LOG("1) Algorithm Method : Non-hogel based hologram generation from Light Field\n");
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
	);
	LOG("3) Random Phase Use : %s\n", GetRandomPhase() ? "Y" : "N");
	LOG("4) Number of Images : %d x %d\n", num_image[_X], num_image[_Y]);

	auto begin = CUR_TIME;
	if (m_mode & MODE_GPU)
	{
		LOG("Not implement GPU version");
	}
	else
	{
		setBuffer();
		makePlaneWaveWField(thetaX, thetaY);
		convertLF2ComplexFieldUsingNonHogelMethod();
		Real distance = 0.0;
		for (uint ch = 0; ch < context_.waveNum; ch++)
			Fresnel_FFT(Hologram, complex_H[ch], context_.wave_length[ch], distance);
			//fresnelPropagation(Hologram, complex_H[ch], distance, ch); //distanceRS2Holo
	}

	LOG("Total Elapsed Time: %.5lf (sec)\n", ELAPSED_TIME(begin, CUR_TIME));
}

//int ophLF::saveAsOhc(const char * fname)
//{
//	setPixelNumberOHC(getEncodeSize());
//
//	Openholo::saveAsOhc(fname);
//
//	return 0;
//}

void ophNonHogelLF::setBuffer()
{
	const uint nU = num_image[_X];
	const uint nV = num_image[_Y];
	nBufferX = (nU >> 1) + 1;
	nBufferY = (nV >> 1) + 1;
}

void ophNonHogelLF::initializeLF()
{
	if (LF != nullptr) {
		for (int i = 0; i < nImages; i++) {
			if (LF[i]) {
				delete[] LF[i];
				LF[i] = nullptr;
			}
		}
		delete[] LF;
		LF = nullptr;
	}

	LF = new uchar*[num_image[_X] * num_image[_Y]];
	for (int i = 0; i < num_image[_X] * num_image[_Y]; i++) {
		LF[i] = new uchar[resolution_image[_X] * resolution_image[_Y]];
		memset(LF[i], 0, resolution_image[_X] * resolution_image[_Y]);
	}
	nImages = num_image[_X] * num_image[_Y];
}

void ophNonHogelLF::makeRandomWField()
{
	auto begin = CUR_TIME;

	const uint nU = num_image[_X];
	const uint nV = num_image[_Y];
	const uint nUV = nU * nV;
	const uint nX = resolution_image[_X];
	const uint nY = resolution_image[_Y];
	const uint nXY = nX * nY;
	const uint nXwithBuffer = nX + 2 * nBufferX;
	const uint nYwithBuffer = nY + 2 * nBufferY;
	const long long int nXYwithBuffer = nXwithBuffer * nYwithBuffer;

	if (WField) {
		delete[] WField;
		WField = nullptr;
	}
	WField = new Complex<Real>[nXYwithBuffer];
	memset(WField, 0.0, sizeof(Complex<Real>) * nXYwithBuffer);

	Complex<Real> phase(0.0, 0.0);
	Real randVal;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int idxnXY = 0; idxnXY < nXYwithBuffer; idxnXY++) {
		randVal = rand((Real)0, (Real)1, idxnXY);
		phase(0.0, 2.0 * M_PI * randVal); //randVal
		WField[idxnXY] = exp(phase);
	}
	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

void ophNonHogelLF::makePlaneWaveWField(double thetaX, double thetaY)
{
	auto begin = CUR_TIME;
	const uint nU = num_image[_X];
	const uint nV = num_image[_Y];
	const uint nUV = nU * nV;
	const uint nX = resolution_image[_X];
	const uint nY = resolution_image[_Y];
	const uint nXY = nX * nY;
	const long long int nXwithBuffer = nX + 2 * nBufferX;
	const long long int nYwithBuffer = nY + 2 * nBufferY;
	const long long int nXYwithBuffer = nXwithBuffer * nYwithBuffer;
	const double px = context_.pixel_pitch[_X];
	const double py = context_.pixel_pitch[_Y];


	if (WField) {
		delete[] WField;
		WField = nullptr;
	}
	WField = new Complex<Real>[nXYwithBuffer];
	memset(WField, 0.0, sizeof(Complex<Real>) * nXYwithBuffer);
	
	Complex<Real> phase(0.0, 0.0);
	double carrierFreqX = (1.0 / context_.wave_length[0])*sin(thetaX);
	double carrierFreqY = (1.0 / context_.wave_length[0])*sin(thetaY);
	for (int idxnX = 0; idxnX < nXwithBuffer; idxnX++) {
		for (int idxnY = 0; idxnY < nYwithBuffer; idxnY++) {
			phase(0.0, 2.0*M_PI* (carrierFreqX*idxnX*px + carrierFreqY*idxnY*py));
			WField[idxnX + nXwithBuffer*idxnY] = exp(phase);
		}
	}
	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

void ophNonHogelLF::fourierTransformOverUVLF()
{
	auto begin = CUR_TIME;
	const int nX = resolution_image[_X];	// resolution of each orthographic images = spatial resolution of LF
	const int nY = resolution_image[_Y];  
	const long long int nXY = nX * nY;
	const int nU = num_image[_X];			// number of orthographic images = angular resolution of LF
	const int nV = num_image[_Y];
	const int nUV = nU * nV;
	const long long int nXwithBuffer = nX + 2 * nBufferX;
	const long long int  nYwithBuffer = nY + 2 * nBufferY;
	const long long int nXYwithBuffer = nXwithBuffer * nYwithBuffer;

	
	// initialize
	if (FToverUV_LF) {
		for (int idxnUV = 0; idxnUV < nUV; idxnUV++) {
			if (FToverUV_LF[idxnUV]) {
				delete[] FToverUV_LF[idxnUV];
				FToverUV_LF[idxnUV] = nullptr;
			}
		}
		delete[] FToverUV_LF;
		FToverUV_LF = nullptr;
	}

	FToverUV_LF = new Complex<Real>*[nUV];
	for (int idxnUV = 0; idxnUV < nUV; idxnUV++) {
		FToverUV_LF[idxnUV] = new Complex<Real>[nXY];
		memset(FToverUV_LF[idxnUV], 0.0, nXY);
	}

	// fft over uv axis
	Complex<Real>* LFatXY = new Complex<Real>[nUV];
	Complex<Real>* FToverUVatXY = new Complex<Real>[nUV];
	int progressCheckPoint = 10000;
	int idxProgress = 0;
	for (int idxnXY = 0; idxnXY < nXY; idxnXY++) {
		memset(LFatXY, 0.0, sizeof(Complex<Real>) * nUV);
		memset(FToverUVatXY, 0.0, sizeof(Complex<Real>) * nUV);

		for (int idxnUV = 0; idxnUV < nUV; idxnUV++) {
			LFatXY[idxnUV] = LF[idxnUV][idxnXY];
		}
		
		fft2(num_image, LFatXY, OPH_FORWARD, OPH_ESTIMATE); // num_image ==> ivec2(nU,nV)
		fft2(LFatXY, FToverUVatXY, nU, nV, OPH_FORWARD);
		fftFree();
		for (int idxnUV = 0; idxnUV < nUV; idxnUV++) {
			FToverUV_LF[idxnUV][idxnXY] = FToverUVatXY[idxnUV];
		}
		
		// only for debugging
		if (idxProgress == progressCheckPoint ) {
			LOG("idxnXY : %1d out of nXY= %llu\n", idxnXY, nXY);
			idxProgress = 0;
		}
		else {
			idxProgress++;
		}
	}
	delete[] LFatXY;
	delete[] FToverUVatXY;
	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));

}

void ophNonHogelLF::convertLF2ComplexFieldUsingNonHogelMethod()
{
	auto begin = CUR_TIME;

	const int nX = resolution_image[_X];	// resolution of each orthographic images = spatial resolution of LF
	const int nY = resolution_image[_Y];
	const long long int nXY = nX * nY;
	const int nU = num_image[_X];			// number of orthographic images = angular resolution of LF
	const int nV = num_image[_Y];
	const int nUV = nU * nV;
	const long long int nXwithBuffer = nX + 2 * nBufferX;
	const long long int nYwithBuffer = nY + 2 * nBufferY;
	const long long int nXYwithBuffer = nXwithBuffer * nYwithBuffer;

	if (Hologram) {
		delete[] Hologram;
		Hologram = nullptr;
	}
	Hologram = new Complex<Real>[nXY];
	memset(Hologram, 0.0, sizeof(Complex<Real>) * nXY);

	Complex<Real>* HologramWithBuffer = new Complex<Real>[nXYwithBuffer];
	memset(HologramWithBuffer, 0.0, sizeof(Complex<Real>) * nXYwithBuffer);

	int startXH = 0;
	int startXW = 0;
	int startYH = 0;
	int startYW = 0;

	int idxnU = 0;
	int idxnV = 0;
	int idxnX = 0;
	int idxnY = 0;
	for (idxnU = 0; idxnU < nU; idxnU++) {
		startXH = (int)((((double)idxnU) - 1.) / 2.) + (int)(-(((double)nU) + 1.) / 4. - ((double)(nX))/2.0 + ((double)(nXwithBuffer)) / 2.0);
		startXW = startXH + (int)(((double)nU) / 2.) - (idxnU - 1);

		for (idxnV = 0; idxnV < nV; idxnV++) {
			startYH = (int)((((double)idxnV) - 1.) / 2.) + (int)(-(((double)nV) + 1.) / 4. - ((double)(nY)) / 2.0 + ((double)(nYwithBuffer)) / 2.0);
			startYW = startYH + (int)(((double)nV) / 2.) - (idxnV - 1);

			for (idxnX = 0; idxnX < nX; idxnX++) {
				for (idxnY = 0; idxnY < nY; idxnY++) {

					HologramWithBuffer[(startXH + idxnX) + nXwithBuffer*(startYH + idxnY)] += FToverUV_LF[idxnU + nU*idxnV][idxnX+nX*idxnY] * WField[(startXW + idxnX) + nXwithBuffer*(startYW + idxnY)];
					
				}
			}
		}
	}

	for (idxnX = 0; idxnX < nX; idxnX++) {
		for (idxnY = 0; idxnY < nY; idxnY++) {
			Hologram[idxnX + nX*idxnY] = HologramWithBuffer[(nBufferX + idxnX) + nXwithBuffer*(nBufferY + idxnY)];
		}
	}
	delete[] HologramWithBuffer;
	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

void ophNonHogelLF::writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, Complex<Real>* complexvalue, int k)
{
	const int n = nx * ny;

	double* intensity = (double*)malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
		intensity[i] = complexvalue[i].real();
	//intensity[i] = complexvalue[i].mag2();

	double min_val, max_val;
	min_val = intensity[0];
	max_val = intensity[0];

	for (int i = 0; i < n; ++i)
	{
		if (min_val > intensity[i])
			min_val = intensity[i];
		else if (max_val < intensity[i])
			max_val = intensity[i];
	}

	char fname[100];
	strcpy(fname, fileName);
	if (k != -1)
	{
		char num[30];
		sprintf(num, "_%d", k);
		strcat(fname, num);
	}
	strcat(fname, ".bmp");

	//LOG("minval %e, max val %e\n", min_val, max_val);

	unsigned char* cgh = (unsigned char*)malloc(sizeof(unsigned char)*n);

	for (int i = 0; i < n; ++i) {
		double val = (intensity[i] - min_val) / (max_val - min_val);
		//val = pow(val, 1.0 / 1.5);
		val = val * 255.0;
		unsigned char v = (uchar)val;

		cgh[i] = v;
	}

	int ret = Openholo::saveAsImg(fname, 8, cgh, nx, ny);

	free(intensity);
	free(cgh);
}