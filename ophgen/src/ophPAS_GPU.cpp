#define OPH_DM_EXPORT 

#include "ophPAS_GPU.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <cooperative_groups.h>
#include "sys.h"
#include <cuda.h>

#include <cuda_device_runtime_api.h>
#include "tinyxml2.h"
#include "PLYparser.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_constants.h>

//CGHEnvironmentData CONF;	// config

using namespace std;

ophPAS_GPU::ophPAS_GPU(void)
	: ophGen()
{
}

ophPAS_GPU::~ophPAS_GPU()
{

}


/**
@fn bool readConfig(const char* name)
@brief config xml파일 파싱 함수
@return 성공시 true, 실패시 false
@param fname: config xml 파일 이름

*/
bool ophPAS_GPU::readConfig(const char* fname) {
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
	LOG("%d", ret);
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

	int nWave = 1;
	auto next = xml_node->FirstChildElement("ScaleX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&pc_config.scale[_X]))
		return false;
	next = xml_node->FirstChildElement("ScaleY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&pc_config.scale[_Y]))
		return false;
	next = xml_node->FirstChildElement("ScaleZ");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&pc_config.scale[_Z]))
		return false;
	next = xml_node->FirstChildElement("Distance");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&pc_config.distance))
		return false;




	next = xml_node->FirstChildElement("SLM_WaveNum"); // OffsetInDepth
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&nWave))
		return false;

	context_.waveNum = nWave;
	if (context_.wave_length) delete[] context_.wave_length;
	context_.wave_length = new Real[nWave];

	char szNodeName[32] = { 0, };
	for (int i = 1; i <= nWave; i++) {
		sprintf(szNodeName, "SLM_WaveLength_%d", i);
		next = xml_node->FirstChildElement(szNodeName);
		if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.wave_length[i - 1]))
			return false;
	}
	next = xml_node->FirstChildElement("SLM_PixelNumX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_X]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelNumY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_Y]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelPitchX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_X]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelPitchY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_Y]))
		return false;
	next = xml_node->FirstChildElement("IMG_Rotation");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryBoolText(&imgCfg.rotate))
		imgCfg.rotate = false;
	next = xml_node->FirstChildElement("IMG_Merge");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryBoolText(&imgCfg.merge))
		imgCfg.merge = false;
	next = xml_node->FirstChildElement("IMG_Flip");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&imgCfg.flip))
		imgCfg.flip = 0;
	next = xml_node->FirstChildElement("DoublePrecision");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryBoolText(&context_.bUseDP))
		context_.bUseDP = true;
	next = xml_node->FirstChildElement("ShiftX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.shift[_X]))
		context_.shift[_X] = 0.0;
	next = xml_node->FirstChildElement("ShiftY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.shift[_Y]))
		context_.shift[_Y] = 0.0;
	next = xml_node->FirstChildElement("ShiftZ");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.shift[_Z]))
		context_.shift[_Z] = 0.0;
	next = xml_node->FirstChildElement("FieldLength");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_dFieldLength))
		m_dFieldLength = 0.0;
	next = xml_node->FirstChildElement("NumOfStream");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&m_nStream))
		m_nStream = 1;

	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	Openholo::setPixelNumberOHC(context_.pixel_number);
	Openholo::setPixelPitchOHC(context_.pixel_pitch);

	OHC_encoder->clearWavelength();
	for (int i = 0; i < nWave; i++)
		Openholo::setWavelengthOHC(context_.wave_length[i], LenUnit::m);

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);
	initialize();
	return true;
}

/**
@fn bool loadPoint(const char* name)
@brief config xml파일 파싱 함수
@return 성공시 true, 실패시 false
@param fname: config xml 파일 이름

*/
int ophPAS_GPU::loadPoint(const char* _filename)
{
	n_points = ophGen::loadPointCloud(_filename, &pc_data);
	return n_points;
}




int ophPAS_GPU::save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py)
{
	if (fname == nullptr) return -1;

	uchar* source = src;
	ivec2 p(px, py);

	if (src == nullptr)
		source = m_lpNormalized[0];
	if (px == 0 && py == 0)
		p = ivec2(context_.pixel_number[_X], context_.pixel_number[_Y]);

	if (checkExtension(fname, ".bmp")) 	// when the extension is bmp
		return Openholo::saveAsImg(fname, bitsperpixel, source, p[_X], p[_Y]);
	else {									// when extension is not .ohf, .bmp - force bmp
		char buf[256];
		memset(buf, 0x00, sizeof(char) * 256);
		sprintf(buf, "%s.bmp", fname);

		return Openholo::saveAsImg(buf, bitsperpixel, source, p[_X], p[_Y]);
	}
}

/**
@fn void save(const char* fname)
@brief 이미지 저장함수
@return
@param fname: 저장할 이미지 파일이름
*/
void ophPAS_GPU::save(const char * fname)
{
	save(fname, 8, cgh_fringe, context_.pixel_number[_X], context_.pixel_number[_Y]);
	delete[] cgh_fringe;
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
char* ophPAS_GPU::rtrim(char* s)
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
char* ophPAS_GPU::ltrim(char* s)
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
char* ophPAS_GPU::trim(char* s)
{
	return rtrim(ltrim(s));
}


/**
@fn void DataInit(OphPointCloudConfig &conf)
@brief PAS알고리즘 수행을 위한 데이터 초기화 함수
@return 없음
@param
conf: OpenHolo Config를 위한 구조체
*/
void ophPAS_GPU::DataInit(OphPointCloudConfig &conf)
{
	m_pHologram = new double[getContext().pixel_number[_X] * getContext().pixel_number[_Y]];
	memset(m_pHologram, 0x00, sizeof(double)*getContext().pixel_number[_X] * getContext().pixel_number[_Y]);

	//for (int i = 0; i < NUMTBL; i++) {
	//	float theta = (float)M2_PI * (float)(i + i - 1) / (float)(2 * NUMTBL);
	//	m_COStbl[i] = (float)cos(theta);
	//	m_SINtbl[i] = (float)sin(theta);
	//}// -> gpu 
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

/**
@fn void PASCalculation(long voxnum, unsigned char * cghfringe, OphPointCloudData *data, OphPointCloudConfig& conf)
@brief PAS알고리즘 수행 함수
@return 없음
@param 
	voxnum: vertex 개수
	cghfringe: 이미지로 저장할 배열
	data: OpenHolo Data관련 구조체
	conf: OpenHolo Config관련 구조체
*/
void ophPAS_GPU::PASCalculation(long voxnum, unsigned char * cghfringe, OphPointCloudData *data, OphPointCloudConfig& conf) {
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
	/*
	고속화 필요한 변수
	m_pHologram
	cghfringe
	변수는 unified memory 사용하는 방향으로
	*/
	

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

/**
@fn void PAS(long voxnum, OphPointCloudData *data, double* m_pHologram,OphPointCloudConfig& conf)
@brief implementation of the PAS
@return 없음
@param
voxnum: vertex 개수
data: OpenHolo Data관련 구조체
m_pHologram: fftw 결과를 넣을 변수
conf: OpenHolo Config관련 구조체
*/
void ophPAS_GPU::PAS(long voxelnum, OphPointCloudData *data, double * m_pHologram, OphPointCloudConfig& conf)
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
	for (no = 0; no < voxelnum * 3; no += 3)
	{
		// point coordinate
		X = (data->vertices[no].point.pos[_X]) * cghScale;
		Y = (data->vertices[no].point.pos[_Y]) * cghScale;
		Z = (data->vertices[no].point.pos[_Z]) * cghScale - defaultDepth;
		Amplitude = data->vertices[no].phase;

		std::cout << "X: " << X << ", Y: " << Y << ", Z: " << Z << ", Amp: " << Amplitude << endl;

		//c_x = X;
		//c_y = Y;
		//c_z = Z;
		//amplitude = Amplitude;
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

/**
@fn void DataInit(int segsize, int cghwidth, int cghheight, float xiinter, float etainter)
@brief PAS 알고리즘 수행을 위한 데이터 초기화 함수
@return 없음

@param 
	segsize:
	cghwidth:
	cghheight:
	xiinter:
	etainter:
*/
void ophPAS_GPU::DataInit(int segsize, int cghwidth, int cghheight, float xiinter, float etainter)
{
	int i, j;
	
	

	// size
	m_segSize = segsize;
	m_hsegSize = (int)(m_segSize / 2);
	m_dsegSize = m_segSize*m_segSize;
	m_segNumx = (int)(cghwidth / m_segSize);
	m_segNumy = (int)(cghheight / m_segSize);
	m_hsegNumx = (int)(m_segNumx / 2);
	m_hsegNumy = (int)(m_segNumy / 2);



	cudaMallocHost((void**)&m_inRe_h, sizeof(float)*m_segNumy * m_segNumx * m_segSize * m_segSize);
	cudaMallocHost((void**)&m_inIm_h, sizeof(float)*m_segNumy * m_segNumx * m_segSize * m_segSize);
	cudaMallocHost((void**)&m_Coefficient_cx, sizeof(int)*m_segNumx);
	cudaMallocHost((void**)&m_Coefficient_cy, sizeof(int)*m_segNumy);
	cudaMallocHost((void**)&m_xc, sizeof(float)*m_segNumx);
	cudaMallocHost((void**)&m_yc, sizeof(float)*m_segNumy);
	cudaMallocHost((void**)&m_COStbl, sizeof(float)*NUMTBL);
	cudaMallocHost((void**)&m_SINtbl, sizeof(float)*NUMTBL);
	
	/*m_inRe_h = new float[m_segNumy * m_segNumx * m_segSize * m_segSize]{ 0 };
	m_inIm_h = new float[m_segNumy * m_segNumx * m_segSize * m_segSize]{ 0 };
	m_COStbl = new float[NUMTBL];
	m_SINtbl = new float[NUMTBL];

	
	m_Coefficient_cx = new int[m_segNumx];
	m_Coefficient_cy = new int[m_segNumy];
	m_xc = new float[m_segNumx];
	m_yc = new float[m_segNumy];*/
	// calculation components
	m_SFrequency_cx = new float[m_segNumx];
	m_SFrequency_cy = new float[m_segNumy];
	m_PickPoint_cx = new int[m_segNumx];
	m_PickPoint_cy = new int[m_segNumy];

	
	for (i = 0; i<NUMTBL; i++) {
		float theta = (float)M2_PI * (float)(i + i - 1) / (float)(2 * NUMTBL);
		m_COStbl[i] = (float)cos(theta);
		m_SINtbl[i] = (float)sin(theta);
	}
	

	

	// base spatial frequency
	m_sf_base = (float)(1.0 / (xiinter*m_segSize));

	
	
	
	

	/*
	for (i = 0; i<m_segNumy; i++) {
		for (j = 0; j<m_segNumx; j++) {
			m_inRe[i*m_segNumx + j] = new float[m_segSize * m_segSize];
			m_inIm[i*m_segNumx + j] = new float[m_segSize * m_segSize];
			memset(m_inRe[i*m_segNumx + j], 0x00, sizeof(float) * m_segSize * m_segSize);
			memset(m_inIm[i*m_segNumx + j], 0x00, sizeof(float) * m_segSize * m_segSize);
		}
	}
	*/
	m_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * m_segSize * m_segSize);
	m_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * m_segSize * m_segSize);
	memset(m_in, 0x00, sizeof(fftw_complex) * m_segSize * m_segSize);

	// segmentation center point calculation
	for (i = 0; i<m_segNumy; i++)
		m_yc[i] = ((i - m_hsegNumy) * m_segSize + m_hsegSize) * etainter;
	for (i = 0; i<m_segNumx; i++)
		m_xc[i] = ((i - m_hsegNumx) * m_segSize + m_hsegSize) * xiinter;

	m_plan = fftw_plan_dft_2d(m_segSize, m_segSize, m_in, m_out, FFTW_BACKWARD, FFTW_ESTIMATE);

	//sex = m_segNumx;
	//sey = m_segNumy;
	//sen = segsize;
	
}

void ophPAS_GPU::MemoryRelease(void)
{
	int i, j;

	cudaFree(&se);

	fftw_destroy_plan(m_plan);
	fftw_free(m_in);
	fftw_free(m_out);

	delete[] m_SFrequency_cx;
	delete[] m_SFrequency_cy;
	delete[] m_PickPoint_cx;
	delete[] m_PickPoint_cy;
	
	/*delete[] m_Coefficient_cx;
	delete[] m_Coefficient_cy;
	delete[] m_xc;
	delete[] m_yc;
	delete[] m_COStbl;
	delete[] m_SINtbl;
	delete[] m_inRe_h;
	delete[] m_inIm_h;*/
	
	cudaFreeHost(m_Coefficient_cx);
	cudaFreeHost(m_Coefficient_cy);
	cudaFreeHost(m_xc);
	cudaFreeHost(m_yc);
	cudaFreeHost(m_COStbl);
	cudaFreeHost(m_SINtbl);
	cudaFreeHost(m_inRe_h);
	cudaFreeHost(m_inIm_h);
	
	/*
	for (i = 0; i<m_segNumy; i++) {
		for (j = 0; j<m_segNumx; j++) {
			delete[] m_inRe[i*m_segNumx + j];
			delete[] m_inIm[i*m_segNumx + j];
		}
	}
	*/
	

}

void ophPAS_GPU::generateHologram()
{

	auto begin = CUR_TIME;
	cgh_fringe = new unsigned char[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	PASCalculation(n_points, cgh_fringe, &pc_data, pc_config);
	auto end = CUR_TIME;
	m_elapsedTime = ((std::chrono::duration<Real>)(end - begin)).count();
	LOG("Total Elapsed Time: %lf (s)\n", m_elapsedTime);
	
}

void ophPAS_GPU::PASCalculation_GPU(long voxnum, unsigned char * cghfringe, OphPointCloudData * data, OphPointCloudConfig & conf)
{

}






void ophPAS_GPU::CalcSpatialFrequency(float cx, float cy, float cz, float amp, int segnumx, int segnumy, int segsize, int hsegsize, float sf_base, float * xc, float * yc, float * sf_cx, float * sf_cy, int * pp_cx, int * pp_cy, int * cf_cx, int * cf_cy, float xiint, float etaint, OphPointCloudConfig& conf)
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



void ophPAS_GPU::CalcCompensatedPhase(float cx, float cy, float cz, float amp
	, int		segNumx, int segNumy
	, int		segsize, int hsegsize, float sf_base
	, float	*xc, float *yc
	, int		*cf_cx, int *cf_cy
	, float	*COStbl, float *SINtbl
	, float	**inRe, float **inIm, OphPointCloudConfig& conf)
{
	/*
	CUDA 처리과정이 순차적으로 처리가 되기 때문에, 호스트에서 디바이스로 데이터를 복사하는 과정에서 GPU는 대기하게 된다.
	따라서 처리과정을 다음과 같이 정한다.
	1. cudaMallocHost를 통해 호스트코드를 강제로 pinned memory로 사용(GPU로 데이터 전송이 빨라질 수 있음)
	2. cudaMemcpyAsync로 데이터 전송 속도 up(cudaMemcpy는 동기화방식)
	3. cudastream 사용으로 병렬처리 극대화
	
	성능비교는 다음과 같이 한다.
	1. 일반적인 CUDA Programming
	2. 개선된  CUDA Programming(pinned memory 사용, cudastream 사용)
	3. CPU 코드

	
	*/


	constValue d_const;
	int num_x = segNumx*segNumy;
	int num_y = segsize*segsize;
	float* inRe_d;
	float* inIm_d;
	

	
	
	


	/*cudaMalloc((void**)&inRe_d, sizeof(float)*num_x*num_y);
	cudaMalloc((void**)&inIm_d, sizeof(float)*num_x*num_y);
	cudaMalloc((void**)&d_const.cf_cx, sizeof(int)*segNumx);
	cudaMalloc((void**)&d_const.cf_cy, sizeof(int)*segNumy);
	cudaMalloc((void**)&d_const.xc, sizeof(float)*segNumx);
	cudaMalloc((void**)&d_const.yc, sizeof(float)*segNumy);
	cudaMalloc((void**)&d_const.costbl, sizeof(float)*NUMTBL);
	cudaMalloc((void**)&d_const.sintbl, sizeof(float)*NUMTBL);*/

	cudaMalloc((void**)&inRe_d, sizeof(float)*num_x*num_y);
	cudaMalloc((void**)&inIm_d, sizeof(float)*num_x*num_y);
	cudaMalloc((void**)&d_const.cf_cx, sizeof(int)*segNumx);
	cudaMalloc((void**)&d_const.cf_cy, sizeof(int)*segNumy);
	cudaMalloc((void**)&d_const.xc, sizeof(float)*segNumx);
	cudaMalloc((void**)&d_const.yc, sizeof(float)*segNumy);
	cudaMalloc((void**)&d_const.costbl, sizeof(float)*NUMTBL);
	cudaMalloc((void**)&d_const.sintbl, sizeof(float)*NUMTBL);

	
	cudaMemcpyAsync(inRe_d, m_inRe_h, sizeof(float)*num_x*num_y, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(inIm_d, m_inIm_h, sizeof(float)*num_x*num_y, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_const.cf_cx, cf_cx , sizeof(int)*segNumx, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_const.cf_cy, cf_cy , sizeof(int)*segNumy, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_const.xc, xc, sizeof(float)*segNumx, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_const.yc, yc, sizeof(float)*segNumy, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_const.costbl, COStbl, sizeof(float)*NUMTBL, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_const.sintbl, SINtbl, sizeof(float)*NUMTBL, cudaMemcpyHostToDevice);
	
	
	
	cuda_Wrapper_phaseCalc(inRe_d, inIm_d, d_const, cx, cy, cz, amp, ivec3(segNumx, segNumy, segsize));
	//phaseCalc << <blockSize, gridSize >> >(inRe_d, inIm_d, d_const);
	
	
	
	
	cudaMemcpyAsync(m_inRe_h, inRe_d , sizeof(float)*num_x*num_y, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(m_inIm_h, inIm_d, sizeof(float)*num_x*num_y, cudaMemcpyDeviceToHost);
	

	/*cudaMemcpy(inRe_d, m_inRe_h, sizeof(float)*num_x*num_y, cudaMemcpyHostToDevice);
	cudaMemcpy(inIm_d, m_inIm_h, sizeof(float)*num_x*num_y, cudaMemcpyHostToDevice);
	cudaMemcpy(d_const.cf_cx, cf_cx, sizeof(int)*segNumx, cudaMemcpyHostToDevice);
	cudaMemcpy(d_const.cf_cy, cf_cy, sizeof(int)*segNumy, cudaMemcpyHostToDevice);
	cudaMemcpy(d_const.xc, xc, sizeof(float)*segNumx, cudaMemcpyHostToDevice);
	cudaMemcpy(d_const.yc, yc, sizeof(float)*segNumy, cudaMemcpyHostToDevice);
	cudaMemcpy(d_const.costbl, COStbl, sizeof(float)*NUMTBL, cudaMemcpyHostToDevice);
	cudaMemcpy(d_const.sintbl, SINtbl, sizeof(float)*NUMTBL, cudaMemcpyHostToDevice);*/

	
	
	/*cudaMemcpyAsync(m_inRe_h, inRe_d, sizeof(float)*i*j, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(m_inIm_h, inIm_d, sizeof(float)*i*j, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFreeHost(d_const.cf_cx);
	cudaFreeHost(d_const.cf_cy);
	cudaFreeHost(d_const.xc);
	cudaFreeHost(d_const.yc);
	cudaFreeHost(d_const.costbl);
	cudaFreeHost(d_const.sintbl);
	cudaFreeHost(inRe_d);
	cudaFreeHost(inIm_d);
	*/
	/*for (int i = 0; i < nStreams; i++) 
		cudaStreamDestroy(streams[i]);*/

	cudaFree(d_const.cf_cx);
	cudaFree(d_const.cf_cy);
	cudaFree(d_const.xc);
	cudaFree(d_const.yc);
	cudaFree(d_const.costbl);
	cudaFree(d_const.sintbl);
	cudaFree(inRe_d);
	cudaFree(inIm_d);

	/*cudaFreeHost(cf_cx);
	cudaFreeHost(cf_cy);
	cudaFreeHost(xc);
	cudaFreeHost(yc);
	cudaFreeHost(COStbl);
	cudaFreeHost(SINtbl);
	cudaFreeHost(m_inRe_h);
	cudaFreeHost(m_inIm_h);*/

	
	/*phaseCalc << <blockSize, gridSize >> >(inRe_d, inIm_d, d_const);
	cudaMemcpy(m_inRe_h, inRe_d, sizeof(float)*num_x*num_y, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_inIm_h, inIm_d, sizeof(float)*num_x*num_y, cudaMemcpyDeviceToHost);
	
	cudaFree(d_const.cf_cx);
	cudaFree(d_const.cf_cy);
	cudaFree(d_const.xc);
	cudaFree(d_const.yc);
	cudaFree(d_const.costbl);
	cudaFree(d_const.sintbl);
	cudaFree(inRe_d);
	cudaFree(inIm_d);*/
}


/**
@fn void RunFFTW(int segnumx, int segnumy, int segsize, int hsegsize, float ** inRe, float ** inIm, fftw_complex * in, fftw_complex * out, fftw_plan * plan, double * pHologram, OphPointCloudConfig& conf)
@brief 푸리에 변환 수행 함수
@return 없음
@param
voxnum: vertex 개수
cghfringe: 이미지로 저장할 배열
data: OpenHolo Data관련 구조체
conf: OpenHolo Config관련 구조체
*/
void ophPAS_GPU::RunFFTW(int segnumx, int segnumy, int segsize, int hsegsize, float ** inRe, float ** inIm, fftw_complex * in, fftw_complex * out, fftw_plan * plan, double * pHologram, OphPointCloudConfig& conf)
{
	int		i, j;
	int		segx, segy;			// coordinate in a Segment 
	int		segxx, segyy;

	int cghWidth = getContext().pixel_number[_X];
	int rows = m_segNumy;
	int cols = m_segNumx;
	
	
	for (segy = 0; segy < segnumy; segy++) {
		for (segx = 0; segx < segnumx; segx++) {
			segyy = segy * segnumx + segx;
			memset(in, 0x00, sizeof(fftw_complex) * segsize * segsize);
			for (i = 0; i < segsize; i++) {
				for (j = 0; j < segsize; j++) {
					segxx = i * segsize + j;
					in[i*segsize + j][0] = m_inRe_h[segyy*segsize*segsize+segxx];
					in[i*segsize + j][1] = m_inIm_h[segyy*segsize*segsize+segxx];
					//inIm_h값이 다름(x) 다 같음
					
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


void ophPAS_GPU::encodeHologram(const vec2 band_limit, const vec2 spectrum_shift)
{
	if (complex_H == nullptr) {
		LOG("Not found diffracted data.");
		return;
	}

	LOG("Single Side Band Encoding..");
	const uint nChannel = context_.waveNum;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const long long int pnXY = pnX * pnY;

	m_vecEncodeSize = ivec2(pnX, pnY);
	context_.ss[_X] = pnX * ppX;
	context_.ss[_Y] = pnY * ppY;
	vec2 ss = context_.ss;

	Real cropx = floor(pnX * band_limit[_X]);
	Real cropx1 = cropx - floor(cropx / 2);
	Real cropx2 = cropx1 + cropx - 1;

	Real cropy = floor(pnY * band_limit[_Y]);
	Real cropy1 = cropy - floor(cropy / 2);
	Real cropy2 = cropy1 + cropy - 1;

	Real* x_o = new Real[pnX];
	Real* y_o = new Real[pnY];

	for (int i = 0; i < pnX; i++)
		x_o[i] = (-ss[_X] / 2) + (ppX * i) + (ppX / 2);

	for (int i = 0; i < pnY; i++)
		y_o[i] = (ss[_Y] - ppY) - (ppY * i);

	Real* xx_o = new Real[pnXY];
	Real* yy_o = new Real[pnXY];

	for (int i = 0; i < pnXY; i++)
		xx_o[i] = x_o[i % pnX];


	for (int i = 0; i < pnX; i++)
		for (int j = 0; j < pnY; j++)
			yy_o[i + j * pnX] = y_o[j];

	Complex<Real>* h = new Complex<Real>[pnXY];

	for (uint ch = 0; ch < nChannel; ch++) {
		fft2(complex_H[ch], h, pnX, pnY, OPH_FORWARD);
		fft2(ivec2(pnX, pnY), h, OPH_FORWARD);
		fftExecute(h);
		fft2(h, h, pnX, pnY, OPH_BACKWARD);

		fft2(h, h, pnX, pnY, OPH_FORWARD);
		fft2(ivec2(pnX, pnY), h, OPH_BACKWARD);
		fftExecute(h);
		fft2(h, h, pnX, pnY, OPH_BACKWARD);

		for (int i = 0; i < pnXY; i++) {
			Complex<Real> shift_phase(1.0, 0.0);
			int r = i / pnX;
			int c = i % pnX;

			Real X = (M_PI * xx_o[i] * spectrum_shift[_X]) / ppX;
			Real Y = (M_PI * yy_o[i] * spectrum_shift[_Y]) / ppY;

			shift_phase[_RE] = shift_phase[_RE] * (cos(X) * cos(Y) - sin(X) * sin(Y));

			m_lpEncoded[ch][i] = (h[i] * shift_phase).real();
		}
	}
	delete[] h;
	delete[] x_o;
	delete[] xx_o;
	delete[] y_o;
	delete[] yy_o;

	LOG("Done.\n");
}

/**
@fn void encoding(unsigned int ENCODE_FLAG)
@brief abstract function of ophGen::encoding
@return 없음
@param
	ENCODE_FLAG: 암호화 메서드
*/
void ophPAS_GPU::encoding(unsigned int ENCODE_FLAG)
{
	ophGen::encoding(ENCODE_FLAG);
}





