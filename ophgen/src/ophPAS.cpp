#define OPH_DM_EXPORT 

#include "ophPAS.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <windows.h>

#include "sys.h"

#include "tinyxml2.h"
#include "PLYparser.h"

//CGHEnvironmentData CONF;	// config

using namespace std;

ophPAS::ophPAS():ophGen()
{
}

ophPAS::~ophPAS()
{
	delete[] m_pHologram;
}

int ophPAS::init(const char* _filename, CGHEnvironmentData* _CGHE)
{
	cout << _filename << endl;

	loadConfig("ConfigInfo.ini", _CGHE);

	return 0;
}

bool ophPAS::loadConfig(const char* filename, CGHEnvironmentData* conf)
{
#define MAX_SIZE 1000
	char inputString[MAX_SIZE];

	ifstream inFile(filename);
	if (!(inFile.is_open()))
	{
		cout << "파일을 찾을 수 없습니다." << endl;
		return false;
	}
	else {
		while (!inFile.eof())
		{
			inFile.getline(inputString, MAX_SIZE);

			// 주석 및 빈칸 제거
			if (!(inputString[0] == NULL || (inputString[0] == '#' && inputString[1] == ' ')))
			{
				char* token = NULL;
				char* parameter = NULL;
				token = strtok(inputString, "=");

				// 데이터 받아서 구조체에 저장
				if (strcmp(token, "CGH width ") == 0) {
					token = strtok(NULL, "=");
					conf->CghWidth = atoi(trim(token));
				}
				else if (strcmp(token, "CGH height ") == 0) {
					token = strtok(NULL, "=");
					conf->CghHeight = atoi(trim(token));
				}
				else if (strcmp(token, "Segmentation size ") == 0) {
					token = strtok(NULL, "=");
					conf->SegmentationSize = atoi(trim(token));
				}
				else if (strcmp(token, "FFT segmentation size ") == 0) {
					token = strtok(NULL, "=");
					conf->fftSegmentationSize = atoi(trim(token));
				}
				else if (strcmp(token, "Red wavelength ") == 0) {
					token = strtok(NULL, "=");
					conf->rWaveLength = atof(trim(token));
				}
				else if (strcmp(token, "Tilting angle on x axis ") == 0) {
					token = strtok(NULL, "=");
					conf->ThetaX = atof(trim(token));
				}
				else if (strcmp(token, "Tilting angle on y axis ") == 0) {
					token = strtok(NULL, "=");
					conf->ThetaY = atof(trim(token));
				}
				else if (strcmp(token, "Default depth ") == 0) {
					token = strtok(NULL, "=");
					conf->DefaultDepth = atof(trim(token));
				}
				else if (strcmp(token, "3D point interval on x axis ") == 0) {
					token = strtok(NULL, "=");
					conf->xInterval = atof(trim(token));
				}
				else if (strcmp(token, "3D point interval on y axis ") == 0) {
					token = strtok(NULL, "=");
					conf->yInterval = atof(trim(token));
				}
				else if (strcmp(token, "Hologram interval on xi axis ") == 0) {
					token = strtok(NULL, "=");
					conf->xiInterval = atof(trim(token));
				}
				else if (strcmp(token, "Hologram interval on eta axis ") == 0) {
					token = strtok(NULL, "=");
					conf->etaInterval = atof(trim(token));
				}
				else if (strcmp(token, "CGH scale ") == 0) {
					token = strtok(NULL, "=");
					conf->CGHScale = atof(trim(token));
				}
			}
		}
	}
	inFile.close();
	return false;
}

bool ophPAS::readConfig(const char* fname, OphPointCloudConfig& configdata) {
	LOG("Reading....%s...", fname);

	auto start = CUR_TIME;

	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fname);
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

#if REAL_IS_DOUBLE & true
	auto next = xml_node->FirstChildElement("ScalingXofPointCloud");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&configdata.scale[_X]))
		return false;
	next = xml_node->FirstChildElement("ScalingYofPointCloud");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&configdata.scale[_Y]))
		return false;
	next = xml_node->FirstChildElement("ScalingZofPointCloud");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&configdata.scale[_Z]))
		return false;
	next = xml_node->FirstChildElement("OffsetInDepth");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&configdata.distance))
		return false;
	next = xml_node->FirstChildElement("SLMpixelPitchX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_X]))
		return false;
	next = xml_node->FirstChildElement("SLMpixelPitchY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_Y]))
		return false;
	next = xml_node->FirstChildElement("WavelengthofLaser");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.wave_length[0]))
		return false;
	//(xml_node->FirstChildElement("ScalingXofPointCloud"))->QueryDoubleText(&configdata.scale[_X]);
	//(xml_node->FirstChildElement("ScalingYofPointCloud"))->QueryDoubleText(&configdata.scale[_Y]);
	//(xml_node->FirstChildElement("ScalingZofPointCloud"))->QueryDoubleText(&configdata.scale[_Z]);
	//(xml_node->FirstChildElement("OffsetInDepth"))->QueryDoubleText(&configdata.offset_depth);
	//(xml_node->FirstChildElement("SLMpixelPitchX"))->QueryDoubleText(&context_.pixel_pitch[_X]);
	//(xml_node->FirstChildElement("SLMpixelPitchY"))->QueryDoubleText(&context_.pixel_pitch[_Y]);
	//(xml_node->FirstChildElement("WavelengthofLaser"))->QueryDoubleText(&context_.wave_length[0]);
	//(xml_node->FirstChildElement("BandpassFilterWidthX"))->QueryDoubleText(&configdata.filter_width[_X]);
	//(xml_node->FirstChildElement("BandpassFilterWidthY"))->QueryDoubleText(&configdata.filter_width[_Y]);
	//(xml_node->FirstChildElement("FocalLengthofInputLens"))->QueryDoubleText(&configdata.focal_length_lens_in);
	//(xml_node->FirstChildElement("FocalLengthofOutputLens"))->QueryDoubleText(&configdata.focal_length_lens_out);
	//(xml_node->FirstChildElement("FocalLengthofEyepieceLens"))->QueryDoubleText(&configdata.focal_length_lens_eye_piece);
	//(xml_node->FirstChildElement("TiltAngleX"))->QueryDoubleText(&configdata.tilt_angle[_X]);
	//(xml_node->FirstChildElement("TiltAngleY"))->QueryDoubleText(&configdata.tilt_angle[_Y]);
#else
	auto next = xml_node->FirstChildElement("ScalingXofPointCloud");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&configdata.scale[_X]))
		return false;
	next = xml_node->FirstChildElement("ScalingYofPointCloud");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&configdata.scale[_Y]))
		return false;
	next = xml_node->FirstChildElement("ScalingZofPointCloud");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&configdata.scale[_Z]))
		return false;
	next = xml_node->FirstChildElement("OffsetInDepth");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&configdata.offset_depth))
		return false;
	next = xml_node->FirstChildElement("SLMpixelPitchX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&context_.pixel_pitch[_X]))
		return false;
	next = xml_node->FirstChildElement("SLMpixelPitchY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&context_.pixel_pitch[_Y]))
		return false;
	next = xml_node->FirstChildElement("WavelengthofLaser");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&context_.wave_length[0]))
		return false;
	//(xml_node->FirstChildElement("ScalingXofPointCloud"))->QueryFloatText(&configdata.scale[_X]);
	//(xml_node->FirstChildElement("ScalingYofPointCloud"))->QueryFloatText(&configdata.scale[_Y]);
	//(xml_node->FirstChildElement("ScalingZofPointCloud"))->QueryFloatText(&configdata.scale[_Z]);
	//(xml_node->FirstChildElement("OffsetInDepth"))->QueryFloatText(&configdata.offset_depth);
	//(xml_node->FirstChildElement("SLMpixelPitchX"))->QueryFloatText(&context_.pixel_pitch[_X]);
	//(xml_node->FirstChildElement("SLMpixelPitchY"))->QueryFloatText(&context_.pixel_pitch[_Y]);
	//(xml_node->FirstChildElement("WavelengthofLaser"))->QueryFloatText(&context_.wave_length[0]);
	//(xml_node->FirstChildElement("BandpassFilterWidthX"))->QueryFloatText(&configdata.filter_width[_X]);
	//(xml_node->FirstChildElement("BandpassFilterWidthY"))->QueryFloatText(&configdata.filter_width[_Y]);
	//(xml_node->FirstChildElement("FocalLengthofInputLens"))->QueryFloatText(&configdata.focal_length_lens_in);
	//(xml_node->FirstChildElement("FocalLengthofOutputLens"))->QueryFloatText(&configdata.focal_length_lens_out);
	//(xml_node->FirstChildElement("FocalLengthofEyepieceLens"))->QueryFloatText(&configdata.focal_length_lens_eye_piece);
	//(xml_node->FirstChildElement("TiltAngleX"))->QueryFloatText(&configdata.tilt_angle[_X]);
	//(xml_node->FirstChildElement("TiltAngleY"))->QueryFloatText(&configdata.tilt_angle[_Y]);
#endif
	next = xml_node->FirstChildElement("SLMpixelNumX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_X]))
		return false;
	next = xml_node->FirstChildElement("SLMpixelNumY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_Y]))
		return false;
	//(xml_node->FirstChildElement("SLMpixelNumX"))->QueryIntText(&context_.pixel_number[_X]);
	//(xml_node->FirstChildElement("SLMpixelNumY"))->QueryIntText(&context_.pixel_number[_Y]);
	//configdata.filter_shape_flag = (int8_t*)(xml_node->FirstChildElement("BandpassFilterShape"))->GetText();

	this->

	context_.k = (2 * M_PI) / context_.wave_length[0];
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	Openholo::setPixelNumberOHC(context_.pixel_number);
	Openholo::setPixelPitchOHC(context_.pixel_pitch);
	Openholo::setWavelengthOHC(context_.wave_length[0], LenUnit::m);

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);
	return true;
}

bool ophPAS::loadPoint(const char* _filename, VoxelStruct* h_vox)
{
#define MAX_SIZE 1000
	char inputString[MAX_SIZE];

	ifstream inFile(_filename);
	if (!(inFile.is_open()))
	{
		cout << "포인트 클라우드 파일을 찾을 수 없습니다." << endl;
		return false;
	}
	else
	{
		inFile.getline(inputString, MAX_SIZE);
		int no = 0;
		while (!inFile.eof())
		{
			inFile.getline(inputString, MAX_SIZE);

			if (inputString[0] != NULL)
			{
				char* token = NULL;

				token = strtok(inputString, "\t");
				h_vox[no].num = atoi(token);	// 인덱스

				token = strtok(NULL, "\t");
				h_vox[no].x = atof(token);	// x 좌표

				token = strtok(NULL, "\t");
				h_vox[no].y = atof(token);	// y 좌표

				token = strtok(NULL, "\t");
				h_vox[no].z = atof(token);	// z 좌표

				token = strtok(NULL, "\t");
				h_vox[no].ph = atof(token);	// phase

				token = strtok(NULL, "\t");
				h_vox[no].r = atof(token);	// red

				//token = strtok(NULL, "\t");
				//h_vox[no].g = atof(token);	// green

				//token = strtok(NULL, "\t");
				//h_vox[no].b = atof(token);	// blue

				no++;
			}
		}
	}
	inFile.close();
	return true;
}


bool ophPAS::load_Num_Point(const char* _filename, long* num_point)
{
#define MAX_SIZE 1000
	char inputString[MAX_SIZE];
	ifstream inFile(_filename);
	if (!(inFile.is_open()))
	{
		cout << "포인트 클라우드 파일을 찾을 수 없습니다." << endl;
		return false;
	}
	else 
	{
		inFile.getline(inputString, MAX_SIZE);
		*num_point = atoi(trim(inputString));
		//cout << *num_point << endl;
	}
	inFile.close();
	return true;
}

int ophPAS::save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py)
{
	if (fname == nullptr) return -1;

	uchar* source = src;
	ivec2 p(px, py);

	if (src == nullptr)
		source = holo_normalized[0];
	if (px == 0 && py == 0)
		p = ivec2(context_.pixel_number[_X], context_.pixel_number[_Y]);

	if (checkExtension(fname, ".bmp")) 	// when the extension is bmp
		return Openholo::saveAsImg(fname, bitsperpixel, source, p[_X], p[_Y]);
	else {									// when extension is not .ohf, .bmp - force bmp
		char buf[256];
		memset(buf, 0x00, sizeof(char) * 256);
		sprintf_s(buf, "%s.bmp", fname);

		return Openholo::saveAsImg(buf, bitsperpixel, source, p[_X], p[_Y]);
	}
}
/*
int ophPAS::saveAsImg(const char * fname, uint8_t bitsperpixel, void * src, int pic_width, int pic_height)
{
	LOG("Saving...%s...", fname);
	auto start = CUR_TIME;

	int _width = pic_width, _height = pic_height;

	int _pixelbytesize = _height * _width * bitsperpixel / 8;
	int _filesize = _pixelbytesize + sizeof(bitmap);

	FILE *fp;
	fopen_s(&fp, fname, "wb");
	if (fp == nullptr) return -1;

	bitmap *pbitmap = (bitmap*)calloc(1, sizeof(bitmap));
	memset(pbitmap, 0x00, sizeof(bitmap));

	pbitmap->fileheader.signature[0] = 'B';
	pbitmap->fileheader.signature[1] = 'M';
	pbitmap->fileheader.filesize = _filesize;
	pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);

	for (int i = 0; i < 256; i++) {
		pbitmap->rgbquad[i].rgbBlue = i;
		pbitmap->rgbquad[i].rgbGreen = i;
		pbitmap->rgbquad[i].rgbRed = i;
	}

	pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	pbitmap->bitmapinfoheader.width = _width;
	pbitmap->bitmapinfoheader.height = _height;
	//pbitmap->bitmapinfoheader.planes = _planes;
	pbitmap->bitmapinfoheader.bitsperpixel = bitsperpixel;
	//pbitmap->bitmapinfoheader.compression = _compression;
	pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	//pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter;
	//pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter;
	pbitmap->bitmapinfoheader.numcolorspallette = 256;
	fwrite(pbitmap, 1, sizeof(bitmap), fp);

	fwrite(src, 1, _pixelbytesize, fp);
	fclose(fp);
	free(pbitmap);

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);

	return 0;
}
*/

// 문자열 우측 공백문자 삭제 함수
char* ophPAS::rtrim(char* s)
{
	char t[MAX_STR_LEN];
	char *end;

	// Visual C 2003 이하에서는
	// strcpy(t, s);
	// 이렇게 해야 함
	strcpy_s(t, s); // 이것은 Visual C 2005용
	end = t + strlen(t) - 1;
	while (end != t && isspace(*end))
		end--;
	*(end + 1) = '\0';
	s = t;

	return s;
}

// 문자열 좌측 공백문자 삭제 함수
char* ophPAS::ltrim(char* s)
{
	char* begin;
	begin = s;

	while (*begin != '\0') {
		if (isspace(*begin))
			begin++;
		else {
			s = begin;
			break;
		}
	}

	return s;
}


// 문자열 앞뒤 공백 모두 삭제 함수
char* ophPAS::trim(char* s)
{
	return rtrim(ltrim(s));
}

void ophPAS::DataInit(CGHEnvironmentData* _CGHE)
{
	m_pHologram = new double[_CGHE->CghHeight*_CGHE->CghWidth];
	memset(m_pHologram, 0x00, sizeof(double)*_CGHE->CghHeight*_CGHE->CghWidth);

	for (int i = 0; i<NUMTBL; i++) {
		float theta = (float)M2_PI * (float)(i + i - 1) / (float)(2 * NUMTBL);
		m_COStbl[i] = (float)cos(theta);
		m_SINtbl[i] = (float)sin(theta);
	}
}

void ophPAS::DataInit(OphPointCloudConfig &conf)
{
	m_pHologram = new double[getContext().pixel_number[_X] * getContext().pixel_number[_Y]];
	memset(m_pHologram, 0x00, sizeof(double)*getContext().pixel_number[_X] * getContext().pixel_number[_Y]);

	for (int i = 0; i < NUMTBL; i++) {
		float theta = (float)M2_PI * (float)(i + i - 1) / (float)(2 * NUMTBL);
		m_COStbl[i] = (float)cos(theta);
		m_SINtbl[i] = (float)sin(theta);
	}
}
/*
void ophPAS::PASCalcuation(long voxnum, unsigned char * cghfringe, VoxelStruct * h_vox, CGHEnvironmentData * _CGHE)
{
	long i, j;

	double Max = -1E9, Min = 1E9;
	double myBuffer;
	int cghwidth = _CGHE->CghWidth;
	int cghheight = _CGHE->CghHeight;

	DataInit(_CGHE);

	//PAS
	//
	PAS(voxnum, h_vox, m_pHologram, _CGHE);
	//

	for (i = 0; i<cghheight; i++) {
		for (j = 0; j<cghwidth; j++) {
			if (Max < m_pHologram[i*cghwidth + j])	Max = m_pHologram[i*cghwidth + j];
			if (Min > m_pHologram[i*cghwidth + j])	Min = m_pHologram[i*cghwidth + j];
		}
	}

	for (i = 0; i<cghheight; i++) {
		for (j = 0; j<cghwidth; j++) {
			myBuffer = 1.0*(((m_pHologram[i*cghwidth + j] - Min) / (Max - Min))*255. + 0.5);
			if (myBuffer >= 255.0)  cghfringe[i*cghwidth + j] = 255;
			else					cghfringe[i*cghwidth + j] = (unsigned char)(myBuffer);
		}
	}

}
*/
void ophPAS::PASCalcuation(long voxnum, unsigned char * cghfringe, OphPointCloudData *data, OphPointCloudConfig& conf) {
	long i, j;

	double Max = -1E9, Min = 1E9;
	double myBuffer;
	int cghwidth = getContext().pixel_number[_X];
	int cghheight = getContext().pixel_number[_Y];

	//DataInit(_CGHE);
	DataInit(conf);

	//PAS
	//
	//PAS(voxnum, h_vox, m_pHologram, _CGHE);
	PAS(voxnum, data, m_pHologram, conf);
	//

	for (i = 0; i < cghheight; i++) {
		for (j = 0; j < cghwidth; j++) {
			if (Max < m_pHologram[i*cghwidth + j])	Max = m_pHologram[i*cghwidth + j];
			if (Min > m_pHologram[i*cghwidth + j])	Min = m_pHologram[i*cghwidth + j];
		}
	}

	for (i = 0; i < cghheight; i++) {
		for (j = 0; j < cghwidth; j++) {
			myBuffer = 1.0*(((m_pHologram[i*cghwidth + j] - Min) / (Max - Min))*255. + 0.5);
			if (myBuffer >= 255.0)  cghfringe[i*cghwidth + j] = 255;
			else					cghfringe[i*cghwidth + j] = (unsigned char)(myBuffer);
		}
	}
}

/*
void ophPAS::PAS(long voxelnum, VoxelStruct * voxel, double * m_pHologram, CGHEnvironmentData* _CGHE)
{
	float xiInterval = _CGHE->xiInterval;
	float etaInterval = _CGHE->etaInterval;
	float cghScale = _CGHE->CGHScale;
	float defaultDepth = _CGHE->DefaultDepth;

	DataInit(_CGHE->fftSegmentationSize, _CGHE->CghWidth, _CGHE->CghHeight, xiInterval, etaInterval);

	int  no;			// voxel Number


	float	X, Y, Z; ;		// x, y, real distance
	float	Amplitude;
	float	sf_base = 1.0 / (xiInterval*_CGHE->fftSegmentationSize);


	//CString mm;
	clock_t start, finish;
	double  duration;
	start = clock();

	// Iteration according to the point number
	for (no = 0; no<voxelnum; no++)
	{
		// point coordinate
		X = (voxel[no].x) * cghScale;
		Y = (voxel[no].y) * cghScale;
		Z = voxel[no].z * cghScale - defaultDepth;
		Amplitude = voxel[no].r;

		CalcSpatialFrequency(X, Y, Z, Amplitude
			, m_segNumx, m_segNumy
			, m_segSize, m_hsegSize, m_sf_base
			, m_xc, m_yc
			, m_SFrequency_cx, m_SFrequency_cy
			, m_PickPoint_cx, m_PickPoint_cy
			, m_Coefficient_cx, m_Coefficient_cy
			, xiInterval, etaInterval,_CGHE);

		CalcCompensatedPhase(X, Y, Z, Amplitude
			, m_segNumx, m_segNumy
			, m_segSize, m_hsegSize, m_sf_base
			, m_xc, m_yc
			, m_Coefficient_cx, m_Coefficient_cy
			, m_COStbl, m_SINtbl
			, m_inRe, m_inIm,_CGHE);

	}

	RunFFTW(m_segNumx, m_segNumy
		, m_segSize, m_hsegSize
		, m_inRe, m_inIm
		, m_in, m_out
		, &m_plan, m_pHologram,_CGHE);

	finish = clock();

	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	//mm.Format("%f", duration);
	//AfxMessageBox(mm);
	cout << duration << endl;
	MemoryRelease();
}
*/

void ophPAS::PAS(long voxelnum, OphPointCloudData *data, double * m_pHologram, OphPointCloudConfig& conf)
{
	float xiInterval = getContext().pixel_pitch[_X];//_CGHE->xiInterval;
	float etaInterval = getContext().pixel_pitch[_Y];//_CGHE->etaInterval;
	float cghScale = conf.scale[_X];// _CGHE->CGHScale;
	float defaultDepth = conf.distance;//_CGHE->DefaultDepth;

	DataInit(FFT_SEGMENT_SIZE, getContext().pixel_number[_X], getContext().pixel_number[_Y], xiInterval, etaInterval);

	long  no;			// voxel Number


	float	X, Y, Z; ;		// x, y, real distance
	float	Amplitude;
	float	sf_base = 1.0 / (xiInterval* FFT_SEGMENT_SIZE);


	//CString mm;
	clock_t start, finish;
	double  duration;
	start = clock();

	// Iteration according to the point number
	for (no = 0; no < voxelnum*3; no+=3)
	{
		// point coordinate
		X = (data->vertex[no]) * cghScale;
		Y = (data->vertex[no+1]) * cghScale;
		Z = data->vertex[no+2] * cghScale - defaultDepth;
		Amplitude = data->phase[no/3];

		std::cout << "X: " << X << ", Y: " << Y << ", Z: " << Z << ", Amp: " << Amplitude << endl;

		/*
		CalcSpatialFrequency(X, Y, Z, Amplitude
			, m_segNumx, m_segNumy
			, m_segSize, m_hsegSize, m_sf_base
			, m_xc, m_yc
			, m_SFrequency_cx, m_SFrequency_cy
			, m_PickPoint_cx, m_PickPoint_cy
			, m_Coefficient_cx, m_Coefficient_cy
			, xiInterval, etaInterval, _CGHE);
		*/
		CalcSpatialFrequency(X, Y, Z, Amplitude
			, m_segNumx, m_segNumy
			, m_segSize, m_hsegSize, m_sf_base
			, m_xc, m_yc
			, m_SFrequency_cx, m_SFrequency_cy
			, m_PickPoint_cx, m_PickPoint_cy
			, m_Coefficient_cx, m_Coefficient_cy
			, xiInterval, etaInterval, conf);

		/*
		CalcCompensatedPhase(X, Y, Z, Amplitude
			, m_segNumx, m_segNumy
			, m_segSize, m_hsegSize, m_sf_base
			, m_xc, m_yc
			, m_Coefficient_cx, m_Coefficient_cy
			, m_COStbl, m_SINtbl
			, m_inRe, m_inIm, _CGHE);
		*/
		CalcCompensatedPhase(X, Y, Z, Amplitude
			, m_segNumx, m_segNumy
			, m_segSize, m_hsegSize, m_sf_base
			, m_xc, m_yc
			, m_Coefficient_cx, m_Coefficient_cy
			, m_COStbl, m_SINtbl
			, m_inRe, m_inIm, conf);

	}

	/*
	RunFFTW(m_segNumx, m_segNumy
		, m_segSize, m_hsegSize
		, m_inRe, m_inIm
		, m_in, m_out
		, &m_plan, m_pHologram, _CGHE);
	*/
	RunFFTW(m_segNumx, m_segNumy
		, m_segSize, m_hsegSize
		, m_inRe, m_inIm
		, m_in, m_out
		, &m_plan, m_pHologram, conf);

	finish = clock();

	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	//mm.Format("%f", duration);
	//AfxMessageBox(mm);
	cout << duration << endl;
	MemoryRelease();
}

void ophPAS::DataInit(int segsize, int cghwidth, int cghheight, float xiinter, float etainter)
{
	int i, j;
	for (i = 0; i<NUMTBL; i++) {
		float theta = (float)M2_PI * (float)(i + i - 1) / (float)(2 * NUMTBL);
		m_COStbl[i] = (float)cos(theta);
		m_SINtbl[i] = (float)sin(theta);
	}

	// size
	m_segSize = segsize;
	m_hsegSize = (int)(m_segSize / 2);
	m_dsegSize = m_segSize*m_segSize;
	m_segNumx = (int)(cghwidth / m_segSize);
	m_segNumy = (int)(cghheight / m_segSize);
	m_hsegNumx = (int)(m_segNumx / 2);
	m_hsegNumy = (int)(m_segNumy / 2);

	// calculation components
	m_SFrequency_cx = new float[m_segNumx];
	m_SFrequency_cy = new float[m_segNumy];
	m_PickPoint_cx = new int[m_segNumx];
	m_PickPoint_cy = new int[m_segNumy];
	m_Coefficient_cx = new int[m_segNumx];
	m_Coefficient_cy = new int[m_segNumy];
	m_xc = new float[m_segNumx];
	m_yc = new float[m_segNumy];

	// base spatial frequency
	m_sf_base = (float)(1.0 / (xiinter*m_segSize));

	m_inRe = new float *[m_segNumy * m_segNumx];
	m_inIm = new float *[m_segNumy * m_segNumx];
	for (i = 0; i<m_segNumy; i++) {
		for (j = 0; j<m_segNumx; j++) {
			m_inRe[i*m_segNumx + j] = new float[m_segSize * m_segSize];
			m_inIm[i*m_segNumx + j] = new float[m_segSize * m_segSize];
			memset(m_inRe[i*m_segNumx + j], 0x00, sizeof(float) * m_segSize * m_segSize);
			memset(m_inIm[i*m_segNumx + j], 0x00, sizeof(float) * m_segSize * m_segSize);
		}
	}

	m_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * m_segSize * m_segSize);
	m_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * m_segSize * m_segSize);
	memset(m_in, 0x00, sizeof(fftw_complex) * m_segSize * m_segSize);

	// segmentation center point calculation
	for (i = 0; i<m_segNumy; i++)
		m_yc[i] = ((i - m_hsegNumy) * m_segSize + m_hsegSize) * etainter;
	for (i = 0; i<m_segNumx; i++)
		m_xc[i] = ((i - m_hsegNumx) * m_segSize + m_hsegSize) * xiinter;

	m_plan = fftw_plan_dft_2d(m_segSize, m_segSize, m_in, m_out, FFTW_BACKWARD, FFTW_ESTIMATE);
}

void ophPAS::MemoryRelease(void)
{
	int i, j;

	fftw_destroy_plan(m_plan);
	fftw_free(m_in);
	fftw_free(m_out);

	delete[] m_SFrequency_cx;
	delete[] m_SFrequency_cy;
	delete[] m_PickPoint_cx;
	delete[] m_PickPoint_cy;
	delete[] m_Coefficient_cx;
	delete[] m_Coefficient_cy;
	delete[] m_xc;
	delete[] m_yc;

	for (i = 0; i<m_segNumy; i++) {
		for (j = 0; j<m_segNumx; j++) {
			delete[] m_inRe[i*m_segNumx + j];
			delete[] m_inIm[i*m_segNumx + j];
		}
	}
	delete[] m_inRe;
	delete[] m_inIm;

}

void ophPAS::CalcSpatialFrequency(float cx, float cy, float cz, float amp, int segnumx, int segnumy, int segsize, int hsegsize, float sf_base, float * xc, float * yc, float * sf_cx, float * sf_cy, int * pp_cx, int * pp_cy, int * cf_cx, int * cf_cy, float xiint, float etaint, CGHEnvironmentData* _CGHE)
{
	int segx, segy;			// coordinate in a Segment 
	float theta_cx, theta_cy;

	float rWaveLength = _CGHE->rWaveLength;
	float thetaX = _CGHE->ThetaX;
	float thetaY = _CGHE->ThetaY;

	for (segx = 0; segx<segnumx; segx++)
	{
		theta_cx = (xc[segx] - cx) / cz;
		sf_cx[segx] = (float)((theta_cx + thetaX) / rWaveLength);
		(sf_cx[segx] >= 0) ? pp_cx[segx] = (int)(sf_cx[segx] / sf_base + 0.5)
			: pp_cx[segx] = (int)(sf_cx[segx] / sf_base - 0.5);
		(abs(pp_cx[segx])<hsegsize) ? cf_cx[segx] = ((segsize - pp_cx[segx]) % segsize)
			: cf_cx[segx] = 0;
	}

	for (segy = 0; segy<segnumy; segy++)
	{
		theta_cy = (yc[segy] - cy) / cz;
		sf_cy[segy] = (float)((theta_cy + thetaY) / rWaveLength);
		(sf_cy[segy] >= 0) ? pp_cy[segy] = (int)(sf_cy[segy] / sf_base + 0.5)
			: pp_cy[segy] = (int)(sf_cy[segy] / sf_base - 0.5);
		(abs(pp_cy[segy])<hsegsize) ? cf_cy[segy] = ((segsize - pp_cy[segy]) % segsize)
			: cf_cy[segy] = 0;
	}
}

void ophPAS::CalcSpatialFrequency(float cx, float cy, float cz, float amp, int segnumx, int segnumy, int segsize, int hsegsize, float sf_base, float * xc, float * yc, float * sf_cx, float * sf_cy, int * pp_cx, int * pp_cy, int * cf_cx, int * cf_cy, float xiint, float etaint, OphPointCloudConfig& conf)
{
	int segx, segy;			// coordinate in a Segment 
	float theta_cx, theta_cy;

	float rWaveLength = getContext().wave_length[0];//_CGHE->rWaveLength;
	float thetaX = 0.0;// _CGHE->ThetaX;
	float thetaY = 0.0;// _CGHE->ThetaY;

	for (segx = 0; segx < segnumx; segx++)
	{
		theta_cx = (xc[segx] - cx) / cz;
		sf_cx[segx] = (float)((theta_cx + thetaX) / rWaveLength);
		(sf_cx[segx] >= 0) ? pp_cx[segx] = (int)(sf_cx[segx] / sf_base + 0.5)
			: pp_cx[segx] = (int)(sf_cx[segx] / sf_base - 0.5);
		(abs(pp_cx[segx]) < hsegsize) ? cf_cx[segx] = ((segsize - pp_cx[segx]) % segsize)
			: cf_cx[segx] = 0;
	}

	for (segy = 0; segy < segnumy; segy++)
	{
		theta_cy = (yc[segy] - cy) / cz;
		sf_cy[segy] = (float)((theta_cy + thetaY) / rWaveLength);
		(sf_cy[segy] >= 0) ? pp_cy[segy] = (int)(sf_cy[segy] / sf_base + 0.5)
			: pp_cy[segy] = (int)(sf_cy[segy] / sf_base - 0.5);
		(abs(pp_cy[segy]) < hsegsize) ? cf_cy[segy] = ((segsize - pp_cy[segy]) % segsize)
			: cf_cy[segy] = 0;
	}
}

void ophPAS::CalcCompensatedPhase(float cx, float cy, float cz, float amp
									, int		segNumx, int segNumy
									, int		segsize, int hsegsize, float sf_base
									, float	*xc, float *yc
									, int		*cf_cx, int *cf_cy
									, float	*COStbl, float *SINtbl
									, float	**inRe, float **inIm, CGHEnvironmentData* _CGHE)
{
	int		segx, segy;			// coordinate in a Segment 
	int		segxx, segyy;
	float	theta_s, theta_c;
	int		dtheta_s, dtheta_c;
	int		idx_c, idx_s;
	float	theta;

	float rWaveNum = _CGHE->rWaveNumber;

	float R;

	for (segy = 0; segy<segNumy; segy++) {
		for (segx = 0; segx<segNumx; segx++) {
			segyy = segy*segNumx + segx;
			segxx = cf_cy[segy] * segsize + cf_cx[segx];
			R = (float)(sqrt((xc[segx] - cx)*(xc[segx] - cx) + (yc[segy] - cy)*(yc[segy] - cy) + cz*cz));
			theta = rWaveNum * R;
			theta_c = theta;
			theta_s = theta + PI;
			dtheta_c = ((int)(theta_c*NUMTBL / M2_PI));
			dtheta_s = ((int)(theta_s*NUMTBL / M2_PI));
			idx_c = (dtheta_c) & (NUMTBL2);
			idx_s = (dtheta_s) & (NUMTBL2);
			inRe[segyy][segxx] += (float)(amp * COStbl[idx_c]);
			inIm[segyy][segxx] += (float)(amp * SINtbl[idx_s]);
		}
	}
}

void ophPAS::CalcCompensatedPhase(float cx, float cy, float cz, float amp
	, int		segNumx, int segNumy
	, int		segsize, int hsegsize, float sf_base
	, float	*xc, float *yc
	, int		*cf_cx, int *cf_cy
	, float	*COStbl, float *SINtbl
	, float	**inRe, float **inIm, OphPointCloudConfig& conf)
{
	int		segx, segy;			// coordinate in a Segment 
	int		segxx, segyy;
	float	theta_s, theta_c;
	int		dtheta_s, dtheta_c;
	int		idx_c, idx_s;
	float	theta;

	float rWaveNum = 9926043.13930423;// _CGHE->rWaveNumber;

	float R;

	for (segy = 0; segy < segNumy; segy++) {
		for (segx = 0; segx < segNumx; segx++) {
			segyy = segy * segNumx + segx;
			segxx = cf_cy[segy] * segsize + cf_cx[segx];
			R = (float)(sqrt((xc[segx] - cx)*(xc[segx] - cx) + (yc[segy] - cy)*(yc[segy] - cy) + cz * cz));
			theta = rWaveNum * R;
			theta_c = theta;
			theta_s = theta + PI;
			dtheta_c = ((int)(theta_c*NUMTBL / M2_PI));
			dtheta_s = ((int)(theta_s*NUMTBL / M2_PI));
			idx_c = (dtheta_c) & (NUMTBL2);
			idx_s = (dtheta_s) & (NUMTBL2);
			inRe[segyy][segxx] += (float)(amp * COStbl[idx_c]);
			inIm[segyy][segxx] += (float)(amp * SINtbl[idx_s]);
		}
	}
}

void ophPAS::RunFFTW(int segnumx, int segnumy, int segsize, int hsegsize, float ** inRe, float ** inIm, fftw_complex * in, fftw_complex * out, fftw_plan * plan, double * pHologram, CGHEnvironmentData* _CGHE)
{
	int		i, j;
	int		segx, segy;			// coordinate in a Segment 
	int		segxx, segyy;

	int cghWidth = _CGHE->CghWidth;

	for (segy = 0; segy<segnumy; segy++) {
		for (segx = 0; segx<segnumx; segx++) {
			segyy = segy*segnumx + segx;
			memset(in, 0x00, sizeof(fftw_complex) * segsize * segsize);
			for (i = 0; i <segsize; i++) {
				for (j = 0; j < segsize; j++) {
					segxx = i*segsize + j;
					in[i*segsize + j][0] = inRe[segyy][segxx];
					in[i*segsize + j][1] = inIm[segyy][segxx];
				}
			}
			fftw_execute(*plan);
			for (i = 0; i <segsize; i++) {
				for (j = 0; j < segsize; j++) {
					pHologram[(segy*segsize + i)*cghWidth + (segx*segsize + j)] = out[i * segsize + j][0];// - out[l * SEGSIZE + m][1];
				}
			}
		}
	}
}

void ophPAS::RunFFTW(int segnumx, int segnumy, int segsize, int hsegsize, float ** inRe, float ** inIm, fftw_complex * in, fftw_complex * out, fftw_plan * plan, double * pHologram, OphPointCloudConfig& conf)
{
	int		i, j;
	int		segx, segy;			// coordinate in a Segment 
	int		segxx, segyy;

	int cghWidth = getContext().pixel_number[_X];

	for (segy = 0; segy < segnumy; segy++) {
		for (segx = 0; segx < segnumx; segx++) {
			segyy = segy * segnumx + segx;
			memset(in, 0x00, sizeof(fftw_complex) * segsize * segsize);
			for (i = 0; i < segsize; i++) {
				for (j = 0; j < segsize; j++) {
					segxx = i * segsize + j;
					in[i*segsize + j][0] = inRe[segyy][segxx];
					in[i*segsize + j][1] = inIm[segyy][segxx];
				}
			}
			fftw_execute(*plan);
			for (i = 0; i < segsize; i++) {
				for (j = 0; j < segsize; j++) {
					pHologram[(segy*segsize + i)*cghWidth + (segx*segsize + j)] = out[i * segsize + j][0];// - out[l * SEGSIZE + m][1];
				}
			}
		}
	}
}