#define OPH_DM_EXPORT 

#include "ophACPAS.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include "sys.h"
#include "tinyxml2.h"
#include "PLYparser.h"

using namespace std;

ophACPAS::ophACPAS():ophGen()
{
}

ophACPAS::~ophACPAS()
{
	delete[] m_pHologram;
}

int ophACPAS::init(const char* _filename, CGHEnvironmentData* _CGHE)
{
	cout << _filename << endl;

	loadConfig("ConfigInfo.ini", _CGHE);

	return 0;
}

bool ophACPAS::loadConfig(const char* filename, CGHEnvironmentData* conf)
{
#define MAX_SIZE 1000
	char inputString[MAX_SIZE];

	ifstream inFile(filename);
	if (!(inFile.is_open()))
	{
		LOG("Cannot find file: %s\n", filename);
		return false;
	}
	else {
		while (!inFile.eof())
		{
			inFile.getline(inputString, MAX_SIZE);

			// 주석 및 빈칸 제거
			if (!(inputString[0] == 0 || (inputString[0] == '#' && inputString[1] == ' ')))
			{
				char* token = NULL;
				char* parameter = NULL;

				token = strtok(inputString, "=");

				// 데이터 받아서 구조체에 저장
				if (strcmp(token, "CGH width ") == 0) {
					token = strtok(nullptr, "=");
					conf->CghWidth = atoi(trim(token));
				}
				else if (strcmp(token, "CGH height ") == 0) {
					token = strtok(nullptr, "=");
					conf->CghHeight = atoi(trim(token));
				}
				else if (strcmp(token, "Segmentation size ") == 0) {
					token = strtok(nullptr, "=");
					conf->SegmentationSize = atoi(trim(token));
				}
				else if (strcmp(token, "FFT segmentation size ") == 0) {
					token = strtok(nullptr, "=");
					conf->fftSegmentationSize = atoi(trim(token));
				}
				else if (strcmp(token, "Red wavelength ") == 0) {
					token = strtok(nullptr, "=");
					conf->rWaveLength = atof(trim(token));
				}
				else if (strcmp(token, "Tilting angle on x axis ") == 0) {
					token = strtok(nullptr, "=");
					conf->ThetaX = atof(trim(token));
				}
				else if (strcmp(token, "Tilting angle on y axis ") == 0) {
					token = strtok(nullptr, "=");
					conf->ThetaY = atof(trim(token));
				}
				else if (strcmp(token, "Default depth ") == 0) {
					token = strtok(nullptr, "=");
					conf->DefaultDepth = atof(trim(token));
				}
				else if (strcmp(token, "3D point interval on x axis ") == 0) {
					token = strtok(nullptr, "=");
					conf->xInterval = atof(trim(token));
				}
				else if (strcmp(token, "3D point interval on y axis ") == 0) {
					token = strtok(nullptr, "=");
					conf->yInterval = atof(trim(token));
				}
				else if (strcmp(token, "Hologram interval on xi axis ") == 0) {
					token = strtok(nullptr, "=");
					conf->xiInterval = atof(trim(token));
				}
				else if (strcmp(token, "Hologram interval on eta axis ") == 0) {
					token = strtok(nullptr, "=");
					conf->etaInterval = atof(trim(token));
				}
				else if (strcmp(token, "CGH scale ") == 0) {
					token = strtok(nullptr, "=");
					conf->CGHScale = atof(trim(token));
				}
			}
		}
	}
	inFile.close();
	return false;
}

bool ophACPAS::readConfig(const char* fname, OphPointCloudConfig& configdata) {
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
#endif
	next = xml_node->FirstChildElement("SLMpixelNumX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_X]))
		return false;
	next = xml_node->FirstChildElement("SLMpixelNumY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_Y]))
		return false;

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

bool ophACPAS::loadPoint(const char* _filename, VoxelStruct* h_vox)
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

			if (inputString[0] != 0)
			{
				char* token = nullptr;

				token = strtok(inputString, "\t");
				h_vox[no].num = atoi(token);	// 인덱스

				token = strtok(nullptr, "\t");
				h_vox[no].x = atof(token);	// x 좌표

				token = strtok(nullptr, "\t");
				h_vox[no].y = atof(token);	// y 좌표

				token = strtok(nullptr, "\t");
				h_vox[no].z = atof(token);	// z 좌표

				token = strtok(nullptr, "\t");
				h_vox[no].ph = atof(token);	// phase

				token = strtok(nullptr, "\t");
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

bool ophACPAS::load_Num_Point(const char* _filename, long* num_point)
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

/*
int ophACPAS::saveAsImg(const char * fname, uint8_t bitsperpixel, void * src, int pic_width, int pic_height)
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

int ophACPAS::save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py)
{
	if (fname == nullptr) return -1;

	uchar* source = src;
	ivec2 p(px, py);
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;

	for (uint ch = 0; ch < nChannel; ch++) {
		if (src == nullptr)
			source = m_lpNormalized[ch];
		if (px == 0 && py == 0)
			p = ivec2(pnX, pnY);

		if (checkExtension(fname, ".bmp")) 	// when the extension is bmp
			return Openholo::saveAsImg(fname, bitsperpixel, source, p[_X], p[_Y]) ? 1 : -1;
		else {									// when extension is not .ohf, .bmp - force bmp
			char buf[256];
			memset(buf, 0x00, sizeof(char) * 256);
			sprintf(buf, "%s.bmp", fname);

			return Openholo::saveAsImg(buf, bitsperpixel, source, p[_X], p[_Y]) ? 1 : -1;
		}
	}
	return -1;
}

// 문자열 우측 공백문자 삭제 함수
char* ophACPAS::rtrim(char* s)
{
	char t[MAX_STR_LEN];
	char *end;

	// Visual C 2003 이하에서는
	// strcpy(t, s);
	// 이렇게 해야 함
	strcpy(t, s); // 이것은 Visual C 2005용
	end = t + strlen(t) - 1;
	while (end != t && isspace(*end))
		end--;
	*(end + 1) = '\0';
	s = t;

	return s;
}

// 문자열 좌측 공백문자 삭제 함수
char* ophACPAS::ltrim(char* s)
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
char* ophACPAS::trim(char* s)
{
	return rtrim(ltrim(s));
}

void ophACPAS::DataInit(CGHEnvironmentData* _CGHE)
{
	m_pHologram = new double[_CGHE->CghHeight*_CGHE->CghWidth];
	memset(m_pHologram, 0x00, sizeof(double)*_CGHE->CghHeight*_CGHE->CghWidth);

	for (int i = 0; i<NUMTBL; i++) {
		float theta = (float)M2_PI * (float)(i + i - 1) / (float)(2 * NUMTBL);
		m_COStbl[i] = (float)cos(theta);
		m_SINtbl[i] = (float)sin(theta);
	}
}

void ophACPAS::DataInit(OphPointCloudConfig& conf)
{
	m_pHologram = new double[getContext().pixel_number[_X] * getContext().pixel_number[_Y]];
	memset(m_pHologram, 0x00, sizeof(double)*getContext().pixel_number[_X] * getContext().pixel_number[_Y]);

	for (int i = 0; i < NUMTBL; i++) {
		float theta = (float)M2_PI * (float)(i + i - 1) / (float)(2 * NUMTBL);
		m_COStbl[i] = (float)cos(theta);
		m_SINtbl[i] = (float)sin(theta);
	}
}

int ophACPAS::ACPASCalcuation(long voxnum, unsigned char * cghfringe, VoxelStruct * h_vox, CGHEnvironmentData * _CGHE)
{
	long i, j;

	double Max = -1E9, Min = 1E9;
	double myBuffer;
	int cghwidth = _CGHE->CghWidth;
	int cghheight = _CGHE->CghHeight;

	DataInit(_CGHE);

	//
	ACPAS(voxnum, h_vox, _CGHE);
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

	return 0;
}

int ophACPAS::ACPASCalcuation(long voxnum, unsigned char * cghfringe, OphPointCloudData *data, OphPointCloudConfig& conf)
{
	long i, j;

	double Max = -1E9, Min = 1E9;
	double myBuffer;
	int cghwidth = getContext().pixel_number[_X];
	int cghheight = getContext().pixel_number[_Y];

	DataInit(conf);

	//
	//ACPAS(voxnum, h_vox, _CGHE);
	ACPAS(voxnum, data, conf);
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

	return 0;
}

void ophACPAS::ACPAS(long voxelnum, OphPointCloudData *data, OphPointCloudConfig& conf)
{
	long  no;			// voxel Number
	int i, j;
	int segx, segy;			// coordinate in a Segment 
	float R;

	int cghwidth = getContext().pixel_number[_X];
	int cghheight = getContext().pixel_number[_Y];
	float xiInterval = getContext().pixel_pitch[_X];
	float etaInterval = getContext().pixel_pitch[_Y];
	float rLamda = getContext().wave_length[0];
	float rWaveNum = 9926043.13930423f;
	float thetaX = 0.0;
	float thetaY = 0.0;

	int segSize = SEG_SIZE;
	int hsegSize = (int)(segSize / 2);
	int dsegSize = segSize * segSize;

	int segNumx = (int)(cghwidth / segSize);
	int segNumy = (int)(cghheight / segSize);
	int hsegNumx = (int)(segNumx / 2);
	int hsegNumy = (int)(segNumy / 2);

	int FFTsegSize = FFT_SEG_SIZE;
	int FFThsegSize = (int)(FFTsegSize / 2);
	int FFTdsegSize = FFTsegSize * FFTsegSize;

	float	X, Y, Z; ;		// x, y, real distance
	float	theta_cx, theta_cy;
	float	*SFrequency_cx = new float[segNumx];	// Center phase
	float	*SFrequency_cy = new float[segNumy];	// Center phase
	int		*PickPoint_cx = new int[segNumx];
	int		*PickPoint_cy = new int[segNumy];
	int		*Coefficient_cx = new int[segNumx];
	int		*Coefficient_cy = new int[segNumy];
	float	*dPhaseSFy = new float[segNumx];
	float	*dPhaseSFx = new float[segNumy];
	float	*xc = new float[segNumx];
	float	*yc = new float[segNumy];
	float	Amplitude;
	float	phase;
	float	sf_base = 1.0 / (xiInterval*FFTsegSize);
	int		segxx, segyy;
	float	theta;
	float	theta_s, theta_c;
	int		dtheta_s, dtheta_c;
	int		idx_c, idx_s;
	double	*Compensation_cx = new double[segNumx];
	double	*Compensation_cy = new double[segNumy];

	//CString mm;

	fftw_complex *in, *out;
	fftw_plan plan;

	double	**inRe = new double *[segNumy * segNumx];
	double	**inIm = new double *[segNumy * segNumx];
	for (i = 0; i < segNumy; i++) {
		for (j = 0; j < segNumx; j++) {
			inRe[i*segNumx + j] = new double[FFTsegSize * FFTsegSize];
			inIm[i*segNumx + j] = new double[FFTsegSize * FFTsegSize];
			memset(inRe[i*segNumx + j], 0x00, sizeof(double) * FFTsegSize * FFTsegSize);
			memset(inIm[i*segNumx + j], 0x00, sizeof(double) * FFTsegSize * FFTsegSize);
		}
	}

	in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FFTsegSize * FFTsegSize);
	out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FFTsegSize * FFTsegSize);
	memset(in, 0x00, sizeof(fftw_complex) * FFTsegSize * FFTsegSize);
	memset(m_pHologram, 0x00, sizeof(double)*cghwidth*cghheight);

	for (segy = 0; segy < segNumy; segy++)
		yc[segy] = ((segy - hsegNumy) * segSize + hsegSize) * etaInterval;
	for (segx = 0; segx < segNumx; segx++)
		xc[segx] = (((segx - hsegNumx) * segSize) + hsegSize) * xiInterval;

	clock_t start, finish;
	double  duration;
	start = clock();

	// Iteration according to the point number
	for (no = 0; no < voxelnum*3; no+=3)
	{
		// point coordinate
		X = data->vertices[no].point.pos[_X] * conf.scale[_X];
		Y = data->vertices[no].point.pos[_Y] * conf.scale[_X];
		Z = data->vertices[no].point.pos[_Z] * conf.scale[_X] - conf.distance;
		Amplitude = data->vertices[no].color.color[_R];
		phase = data->vertices[no].phase;

		for (segy = 0; segy < segNumy; segy++)
		{
			theta_cy = (yc[segy] - Y) / Z;
			SFrequency_cy[segy] = (theta_cy + thetaY) / rLamda;
			(SFrequency_cy[segy] >= 0) ? PickPoint_cy[segy] = (int)(SFrequency_cy[segy] / sf_base + 0.5)
				: PickPoint_cy[segy] = (int)(SFrequency_cy[segy] / sf_base - 0.5);
			(abs(PickPoint_cy[segy]) < FFThsegSize) ? Coefficient_cy[segy] = ((FFTsegSize - PickPoint_cy[segy]) % FFTsegSize)
				: Coefficient_cy[segy] = 0;
			Compensation_cy[segy] = (float)(2 * PI* ((yc[segy] - Y)*SFrequency_cy[segy] + PickPoint_cy[segy] * sf_base*FFThsegSize*xiInterval));
		}

		for (segx = 0; segx < segNumx; segx++)
		{
			theta_cx = (xc[segx] - X) / Z;
			SFrequency_cx[segx] = (theta_cx + thetaX) / rLamda;
			(SFrequency_cx[segx] >= 0) ? PickPoint_cx[segx] = (int)(SFrequency_cx[segx] / sf_base + 0.5)
				: PickPoint_cx[segx] = (int)(SFrequency_cx[segx] / sf_base - 0.5);
			(abs(PickPoint_cx[segx]) < FFThsegSize) ? Coefficient_cx[segx] = ((FFTsegSize - PickPoint_cx[segx]) % FFTsegSize)
				: Coefficient_cx[segx] = 0;
			Compensation_cx[segx] = (float)(2 * PI* ((xc[segx] - X)*SFrequency_cx[segx] + PickPoint_cx[segx] * sf_base*FFThsegSize*etaInterval));
		}
		for (segy = 0; segy < segNumy; segy++) {
			for (segx = 0; segx < segNumx; segx++) {
				segyy = segy * segNumx + segx;
				segxx = Coefficient_cy[segy] * FFTsegSize + Coefficient_cx[segx];
				R = sqrt((xc[segx] - X)*(xc[segx] - X) + (yc[segy] - Y)*(yc[segy] - Y) + Z * Z);
				theta = rWaveNum * R
					+ phase
					+ Compensation_cy[segy] + Compensation_cx[segx];
				//+ dPhaseSFy[segy] + dPhaseSFx[segx];
				theta_c = theta;
				theta_s = theta + PI;
				dtheta_c = ((int)(theta_c*NUMTBL / M2_PI));
				dtheta_s = ((int)(theta_s*NUMTBL / M2_PI));
				idx_c = (dtheta_c) & (NUMTBL2);
				idx_s = (dtheta_s) & (NUMTBL2);
				inRe[segyy][segxx] += (double)(Amplitude * m_COStbl[idx_c]);
				inIm[segyy][segxx] += (double)(Amplitude * m_SINtbl[idx_s]);
			}
		}
	}

	plan = fftw_plan_dft_2d(FFTsegSize, FFTsegSize, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

	for (segy = 0; segy < segNumy; segy++) {
		for (segx = 0; segx < segNumx; segx++) {
			segyy = segy * segNumx + segx;
			memset(in, 0x00, sizeof(fftw_complex) * FFTsegSize * FFTsegSize);
			for (i = 0; i < FFTsegSize; i++) {
				for (j = 0; j < FFTsegSize; j++) {
					segxx = i * FFTsegSize + j;
					in[i*FFTsegSize + j][0] = inRe[segyy][segxx];
					in[i*FFTsegSize + j][1] = inIm[segyy][segxx];
				}
			}
			fftw_execute(plan);
			for (i = 0; i < segSize; i++) {
				for (j = 0; j < segSize; j++) {
					m_pHologram[(segy*segSize + i)*cghwidth + (segx*segSize + j)] +=
						out[(i + FFThsegSize - hsegSize) * FFTsegSize + (j + FFThsegSize - hsegSize)][0];// - out[l * SEGSIZE + m][1];
				}
			}
		}
	}
	finish = clock();

	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	//mm.Format("%f", duration);
	//AfxMessageBox(mm);
	cout << duration << endl;

	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	delete[] dPhaseSFy;
	delete[] dPhaseSFx;
	delete[] SFrequency_cx;
	delete[] SFrequency_cy;
	delete[] PickPoint_cx;
	delete[] PickPoint_cy;
	delete[] Coefficient_cx;
	delete[] Coefficient_cy;
	delete[] xc;
	delete[] yc;
	for (i = 0; i < segNumy; i++) {
		for (j = 0; j < segNumx; j++) {
			delete[] inRe[i*segNumx + j];
			delete[] inIm[i*segNumx + j];
		}
	}
	delete[] inRe;
	delete[] inIm;
}

void ophACPAS::ACPAS(long voxelnum, VoxelStruct * _h_vox, CGHEnvironmentData * _CGHE)
{
	int  no;			// voxel Number
	int i, j;
	int segx, segy;			// coordinate in a Segment 
	float R;

	int cghwidth = _CGHE->CghWidth;
	int cghheight =_CGHE->CghHeight;
	float xiInterval =_CGHE->xiInterval;
	float etaInterval =_CGHE->etaInterval;
	float rLamda =_CGHE->rWaveLength;
	float rWaveNum =_CGHE->rWaveNumber;
	float thetaX =_CGHE->ThetaX;
	float thetaY =_CGHE->ThetaY;

	int segSize =_CGHE->SegmentationSize;
	int hsegSize = (int)(segSize / 2);
	int dsegSize = segSize*segSize;

	int segNumx = (int)(cghwidth / segSize);
	int segNumy = (int)(cghheight / segSize);
	int hsegNumx = (int)(segNumx / 2);
	int hsegNumy = (int)(segNumy / 2);

	int FFTsegSize =_CGHE->fftSegmentationSize;
	int FFThsegSize = (int)(FFTsegSize / 2);
	int FFTdsegSize = FFTsegSize*FFTsegSize;

	float	X, Y, Z; ;		// x, y, real distance
	float	theta_cx, theta_cy;
	float	*SFrequency_cx = new float[segNumx];	// Center phase
	float	*SFrequency_cy = new float[segNumy];	// Center phase
	int		*PickPoint_cx = new int[segNumx];
	int		*PickPoint_cy = new int[segNumy];
	int		*Coefficient_cx = new int[segNumx];
	int		*Coefficient_cy = new int[segNumy];
	float	*dPhaseSFy = new float[segNumx];
	float	*dPhaseSFx = new float[segNumy];
	float	*xc = new float[segNumx];
	float	*yc = new float[segNumy];
	float	Amplitude;
	float	phase;
	float	sf_base = 1.0 / (xiInterval*FFTsegSize);
	int		segxx, segyy;
	float	theta;
	float	theta_s, theta_c;
	int		dtheta_s, dtheta_c;
	int		idx_c, idx_s;
	double	*Compensation_cx = new double[segNumx];
	double	*Compensation_cy = new double[segNumy];

	//CString mm;

	fftw_complex *in, *out;
	fftw_plan plan;

	double	**inRe = new double *[segNumy * segNumx];
	double	**inIm = new double *[segNumy * segNumx];
	for (i = 0; i<segNumy; i++) {
		for (j = 0; j<segNumx; j++) {
			inRe[i*segNumx + j] = new double[FFTsegSize * FFTsegSize];
			inIm[i*segNumx + j] = new double[FFTsegSize * FFTsegSize];
			memset(inRe[i*segNumx + j], 0x00, sizeof(double) * FFTsegSize * FFTsegSize);
			memset(inIm[i*segNumx + j], 0x00, sizeof(double) * FFTsegSize * FFTsegSize);
		}
	}

	in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FFTsegSize * FFTsegSize);
	out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FFTsegSize * FFTsegSize);
	memset(in, 0x00, sizeof(fftw_complex) * FFTsegSize * FFTsegSize);
	memset(m_pHologram, 0x00, sizeof(double)*cghwidth*cghheight);

	for (segy = 0; segy<segNumy; segy++)
		yc[segy] = ((segy - hsegNumy) * segSize + hsegSize) * etaInterval;
	for (segx = 0; segx<segNumx; segx++)
		xc[segx] = (((segx - hsegNumx) * segSize) + hsegSize) * xiInterval;

	clock_t start, finish;
	double  duration;
	start = clock();

	// Iteration according to the point number
	for (no = 0; no<voxelnum; no++)
	{
		// point coordinate
		X = (_h_vox[no].x) *_CGHE->CGHScale;
		Y = (_h_vox[no].y) *_CGHE->CGHScale;
		Z = _h_vox[no].z *_CGHE->CGHScale -_CGHE->DefaultDepth;
		Amplitude = _h_vox[no].r;
		phase = _h_vox[no].ph;

		for (segy = 0; segy<segNumy; segy++)
		{
			theta_cy = (yc[segy] - Y) / Z;
			SFrequency_cy[segy] = (theta_cy + thetaY) / rLamda;
			(SFrequency_cy[segy] >= 0) ? PickPoint_cy[segy] = (int)(SFrequency_cy[segy] / sf_base + 0.5)
				: PickPoint_cy[segy] = (int)(SFrequency_cy[segy] / sf_base - 0.5);
			(abs(PickPoint_cy[segy])<FFThsegSize) ? Coefficient_cy[segy] = ((FFTsegSize - PickPoint_cy[segy]) % FFTsegSize)
				: Coefficient_cy[segy] = 0;
			Compensation_cy[segy] = (float)(2 * PI* ((yc[segy] - Y)*SFrequency_cy[segy] + PickPoint_cy[segy] * sf_base*FFThsegSize*xiInterval));
		}

		for (segx = 0; segx<segNumx; segx++)
		{
			theta_cx = (xc[segx] - X) / Z;
			SFrequency_cx[segx] = (theta_cx + thetaX) / rLamda;
			(SFrequency_cx[segx] >= 0) ? PickPoint_cx[segx] = (int)(SFrequency_cx[segx] / sf_base + 0.5)
				: PickPoint_cx[segx] = (int)(SFrequency_cx[segx] / sf_base - 0.5);
			(abs(PickPoint_cx[segx])<FFThsegSize) ? Coefficient_cx[segx] = ((FFTsegSize - PickPoint_cx[segx]) % FFTsegSize)
				: Coefficient_cx[segx] = 0;
			Compensation_cx[segx] = (float)(2 * PI* ((xc[segx] - X)*SFrequency_cx[segx] + PickPoint_cx[segx] * sf_base*FFThsegSize*etaInterval));
		}
		for (segy = 0; segy<segNumy; segy++) {
			for (segx = 0; segx<segNumx; segx++) {
				segyy = segy*segNumx + segx;
				segxx = Coefficient_cy[segy] * FFTsegSize + Coefficient_cx[segx];
				R = sqrt((xc[segx] - X)*(xc[segx] - X) + (yc[segy] - Y)*(yc[segy] - Y) + Z*Z);
				theta = rWaveNum * R
					+ phase
					+ Compensation_cy[segy] + Compensation_cx[segx];
				//+ dPhaseSFy[segy] + dPhaseSFx[segx];
				theta_c = theta;
				theta_s = theta + PI;
				dtheta_c = ((int)(theta_c*NUMTBL / M2_PI));
				dtheta_s = ((int)(theta_s*NUMTBL / M2_PI));
				idx_c = (dtheta_c) & (NUMTBL2);
				idx_s = (dtheta_s) & (NUMTBL2);
				inRe[segyy][segxx] += (double)(Amplitude * m_COStbl[idx_c]);
				inIm[segyy][segxx] += (double)(Amplitude * m_SINtbl[idx_s]);
			}
		}
	}

	plan = fftw_plan_dft_2d(FFTsegSize, FFTsegSize, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

	for (segy = 0; segy<segNumy; segy++) {
		for (segx = 0; segx<segNumx; segx++) {
			segyy = segy*segNumx + segx;
			memset(in, 0x00, sizeof(fftw_complex) * FFTsegSize * FFTsegSize);
			for (i = 0; i <FFTsegSize; i++) {
				for (j = 0; j < FFTsegSize; j++) {
					segxx = i*FFTsegSize + j;
					in[i*FFTsegSize + j][0] = inRe[segyy][segxx];
					in[i*FFTsegSize + j][1] = inIm[segyy][segxx];
				}
			}
			fftw_execute(plan);
			for (i = 0; i <segSize; i++) {
				for (j = 0; j < segSize; j++) {
					m_pHologram[(segy*segSize + i)*cghwidth + (segx*segSize + j)] +=
						out[(i + FFThsegSize - hsegSize) * FFTsegSize + (j + FFThsegSize - hsegSize)][0];// - out[l * SEGSIZE + m][1];
				}
			}
		}
	}
	finish = clock();

	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	//mm.Format("%f", duration);
	//AfxMessageBox(mm);
	cout << duration << endl;

	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	delete[] dPhaseSFy;
	delete[] dPhaseSFx;
	delete[] SFrequency_cx;
	delete[] SFrequency_cy;
	delete[] PickPoint_cx;
	delete[] PickPoint_cy;
	delete[] Coefficient_cx;
	delete[] Coefficient_cy;
	delete[] xc;
	delete[] yc;
	for (i = 0; i<segNumy; i++) {
		for (j = 0; j<segNumx; j++) {
			delete[] inRe[i*segNumx + j];
			delete[] inIm[i*segNumx + j];
		}
	}
	delete[] inRe;
	delete[] inIm;
}
