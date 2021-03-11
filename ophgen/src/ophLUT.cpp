#define OPH_DM_EXPORT 

#include "ophLUT.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

#include "sys.h"

//CGHEnvironmentData CONF;	// config

using namespace std;

ophLUT::ophLUT():ophGen()
{
}

ophLUT::~ophLUT()
{
	delete[] m_pHologram;
}

int ophLUT::init(const char* _filename, CGHEnvironmentData* _CGHE)
{
	cout << _filename << endl;

	loadConfig("ConfigInfo.ini", _CGHE);

	return 0;
}

bool ophLUT::loadConfig(const char* filename, CGHEnvironmentData* conf)
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
				char* context = nullptr;
				token = strtok_s(inputString, "=", &context);

				// 데이터 받아서 구조체에 저장
				if (strcmp(token, "CGH width ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->CghWidth = atoi(trim(token));
				}
				else if (strcmp(token, "CGH height ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->CghHeight = atoi(trim(token));
				}
				else if (strcmp(token, "Segmentation size ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->SegmentationSize = atoi(trim(token));
				}
				else if (strcmp(token, "FFT segmentation size ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->fftSegmentationSize = atoi(trim(token));
				}
				else if (strcmp(token, "Red wavelength ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->rWaveLength = atof(trim(token));
				}
				else if (strcmp(token, "Tilting angle on x axis ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->ThetaX = atof(trim(token));
				}
				else if (strcmp(token, "Tilting angle on y axis ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->ThetaY = atof(trim(token));
				}
				else if (strcmp(token, "Default depth ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->DefaultDepth = atof(trim(token));
				}
				else if (strcmp(token, "3D point interval on x axis ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->xInterval = atof(trim(token));
				}
				else if (strcmp(token, "3D point interval on y axis ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->yInterval = atof(trim(token));
				}
				else if (strcmp(token, "Hologram interval on xi axis ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->xiInterval = atof(trim(token));
				}
				else if (strcmp(token, "Hologram interval on eta axis ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->etaInterval = atof(trim(token));
				}
				else if (strcmp(token, "CGH scale ") == 0) {
					token = strtok_s(NULL, "=", &context);
					conf->CGHScale = atof(trim(token));
				}
			}
		}
	}
	inFile.close();
	return false;
}

bool ophLUT::loadPoint(const char* _filename, VoxelStruct* h_vox)
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
				char* context;
				token = strtok_s(inputString, "\t", &context);
				h_vox[no].num = atoi(token);	// 인덱스

				token = strtok_s(NULL, "\t", &context);
				h_vox[no].x = atof(token);	// x 좌표

				token = strtok_s(NULL, "\t", &context);
				h_vox[no].y = atof(token);	// y 좌표

				token = strtok_s(NULL, "\t", &context);
				h_vox[no].z = atof(token);	// z 좌표

				token = strtok_s(NULL, "\t", &context);
				h_vox[no].ph = atof(token);	// phase

				token = strtok_s(NULL, "\t", &context);
				h_vox[no].r = atof(token);	// red

				//token = strtok_s(NULL, "\t");
				//h_vox[no].g = atof(token);	// green

				//token = strtok_s(NULL, "\t");
				//h_vox[no].b = atof(token);	// blue

				no++;
			}
		}
	}
	inFile.close();
	return true;
}

bool ophLUT::load_Num_Point(const char* _filename, long* num_point)
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

void ophLUT::DataInit(CGHEnvironmentData* _CGHE)
{
	m_pHologram = new double[_CGHE->CghHeight*_CGHE->CghWidth];
	memset(m_pHologram, 0x00, sizeof(double)*_CGHE->CghHeight*_CGHE->CghWidth);

	for (int i = 0; i<NUMTBL; i++) {
		float theta = (float)M2_PI * (float)(i + i - 1) / (float)(2 * NUMTBL);
		m_COStbl[i] = (float)cos(theta);
		m_SINtbl[i] = (float)sin(theta);
	}
}

// APAS
int ophLUT::LUTCalcuation(long voxnum, unsigned char *cghfringe, VoxelStruct* h_vox, CGHEnvironmentData* _CGHE)
{
	long i, j;

	double Max = -1E9, Min = 1E9;
	double myBuffer;
	int cghwidth = _CGHE->CghWidth;
	int cghheight = _CGHE->CghHeight;

	DataInit(_CGHE);

	//
	APAS(voxnum, h_vox, _CGHE);
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

void ophLUT::APAS(long voxelnum, VoxelStruct * _h_vox, CGHEnvironmentData * _CGHE)
{
	int  no;			// voxel Number
	int i, j;
	int segx, segy;			// coordinate in a Segment 
	float R;

	int cghwidth = _CGHE->CghWidth;
	int cghheight = _CGHE->CghHeight;
	float xiInterval = _CGHE->xiInterval;
	float etaInterval = _CGHE->etaInterval;
	float rLamda = _CGHE->rWaveLength;
	float rWaveNum = _CGHE->rWaveNumber;
	float thetaX = _CGHE->ThetaX;
	float thetaY = _CGHE->ThetaY;

	int segSize = _CGHE->SegmentationSize;
	int hsegSize = (int)(segSize / 2);
	int dsegSize = segSize*segSize;

	int segNumx = (int)(cghwidth / segSize);
	int segNumy = (int)(cghheight / segSize);
	int hsegNumx = (int)(segNumx / 2);
	int hsegNumy = (int)(segNumy / 2);

	int FFTsegSize = _CGHE->fftSegmentationSize;
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
		X = (_h_vox[no].x) * _CGHE->CGHScale;
		Y = (_h_vox[no].y) * _CGHE->CGHScale;
		Z = _h_vox[no].z * _CGHE->CGHScale - _CGHE->DefaultDepth;
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
		}

		for (segx = 0; segx<segNumx; segx++)
		{
			theta_cx = (xc[segx] - X) / Z;
			SFrequency_cx[segx] = (theta_cx + thetaX) / rLamda;
			(SFrequency_cx[segx] >= 0) ? PickPoint_cx[segx] = (int)(SFrequency_cx[segx] / sf_base + 0.5)
				: PickPoint_cx[segx] = (int)(SFrequency_cx[segx] / sf_base - 0.5);
			(abs(PickPoint_cx[segx])<FFThsegSize) ? Coefficient_cx[segx] = ((FFTsegSize - PickPoint_cx[segx]) % FFTsegSize)
				: Coefficient_cx[segx] = 0;
		}
		for (segy = 0; segy<segNumy; segy++) {
			for (segx = 0; segx<segNumx; segx++) {
				segyy = segy*segNumx + segx;
				segxx = Coefficient_cy[segy] * FFTsegSize + Coefficient_cx[segx];
				R = sqrt((xc[segx] - X)*(xc[segx] - X) + (yc[segy] - Y)*(yc[segy] - Y) + Z*Z);
				theta = rWaveNum * R + phase;
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
																										 //					m_pHologram[(segy*segSize+i)*cghwidth+(segx*segSize+j)] += atan2( out[i * FFTsegSize + j][1], out[i * FFTsegSize + j][0]);// - out[l * SEGSIZE + m][1];
				}
			}
		}
	}
	finish = clock();

	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << duration << endl;
	//mm.Format("%f", duration);
	//AfxMessageBox(mm);

	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
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

int ophLUT::saveAsImg(const char * fname, uint8_t bitsperpixel, void * src, int pic_width, int pic_height)
{
	LOG("Saving...%s...", fname);
	auto start = CUR_TIME;

	int _width = pic_width, _height = pic_height;

	int _pixelbytesize = _height * _width * bitsperpixel / 8;
	int _filesize = _pixelbytesize + sizeof(bitmap8bit);

	FILE *fp;
	fopen_s(&fp, fname, "wb");
	if (fp == nullptr) return -1;

	bitmap8bit *pbitmap = (bitmap8bit*)calloc(1, sizeof(bitmap8bit));
	memset(pbitmap, 0x00, sizeof(bitmap8bit));

	pbitmap->fileheader.signature[0] = 'B';
	pbitmap->fileheader.signature[1] = 'M';
	pbitmap->fileheader.filesize = _filesize;
	pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap8bit);

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
	fwrite(pbitmap, 1, sizeof(bitmap8bit), fp);

	fwrite(src, 1, _pixelbytesize, fp);
	fclose(fp);
	free(pbitmap);

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);

	return 0;
}

// 문자열 우측 공백문자 삭제 함수
char* ophLUT::rtrim(char* s)
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
char* ophLUT::ltrim(char* s)
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
char* ophLUT::trim(char* s)
{
	return rtrim(ltrim(s));
}
