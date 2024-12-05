
#include "ophPAS.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

#include "sys.h"

#include "tinyxml2.h"
#include "PLYparser.h"

ophPAS::ophPAS(void)
	: ophGen()
	,coefficient_cx(nullptr)
	,coefficient_cy(nullptr)
	,compensation_cx(nullptr)
	,compensation_cy(nullptr)
	,xc(nullptr)
	,yc(nullptr)
	,input(nullptr)
{
	LOG("*** PHASE-ADDED STEREOGRAM: BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

ophPAS::~ophPAS()
{
	if (coefficient_cx != nullptr)
		delete[] coefficient_cx;
	if (coefficient_cy != nullptr)
		delete[] coefficient_cy;
	if (compensation_cx != nullptr)
		delete[] compensation_cx;
	if (compensation_cy != nullptr)
		delete[] compensation_cy;
	if (xc != nullptr)
		delete[] xc;
	if (yc != nullptr)
		delete[] yc;
}

bool ophPAS::readConfig(const char* fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	bool bRet = true;

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
	sprintf(szNodeName, "ScaleX");
	// about point
	auto next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config.scale[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "ScaleY");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config.scale[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "ScaleZ");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config.scale[_Z]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Distance");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config.distance))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	
	initialize();

	LOG("**************************************************\n");
	LOG("       Read Config (Phase-Added Stereogram)       \n");
	LOG("1) Focal Length : %.5lf\n", pc_config.distance);
	LOG("2) Object Scale : %.5lf / %.5lf / %.5lf\n", pc_config.scale[_X], pc_config.scale[_Y], pc_config.scale[_Z]);
	LOG("**************************************************\n");

	return bRet;
}

int ophPAS::loadPoint(const char* _filename)
{
	int n_points = ophGen::loadPointCloud(_filename, &pc_data);
	return n_points;
}


void ophPAS::generateHologram()
{
	auto begin = CUR_TIME;
	resetBuffer();
	Init();
	CreateLookupTables();
	PAS();
	LOG("Total Elapsed Time: %lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
}


void ophPAS::Init()
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int segSize = is_accurate ? SEG_SIZE : FFT_SEGMENT_SIZE;
	const int snX = pnX / segSize;
	const int snY = pnY / segSize;

	if (coefficient_cx == nullptr)
		coefficient_cx = new int[snX];
	if (coefficient_cy == nullptr)
		coefficient_cy = new int[snY];
	if (compensation_cx == nullptr)
		compensation_cx = new Real[snX];
	if (compensation_cy == nullptr)
		compensation_cy = new Real[snY];
	if (xc == nullptr)
		xc = new Real[snX];
	if (yc == nullptr)
		yc = new Real[snY];
}

void ophPAS::CreateLookupTables()
{
	Real pi2 = M_PI * 2;
	for (int i = 0; i < NUMTBL; i++) {
		Real theta = pi2 * (i + i - 1) / (2 * NUMTBL);
		LUTCos[i] = cos(theta);
		LUTSin[i] = sin(theta);
	}
}

void ophPAS::PAS()
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];

	const int hpnX = pnX >> 1;
	const int hpnY = pnY >> 1;

	const Real distance = pc_config.distance;
	const int N = pnX * pnY;
	const int segSize = is_accurate ? SEG_SIZE : FFT_SEGMENT_SIZE;
	const int hSegSize = segSize >> 1;
	const int sSegSize = segSize * segSize;

	const int fftSegSize = FFT_SEGMENT_SIZE;
	const int ffthSegSize = fftSegSize >> 1;
	const int fftsSegSize = fftSegSize * fftSegSize;

	const int snX = pnX / segSize;
	const int snY = pnY / segSize;
	const int hsnX = snX >> 1;
	const int hsnY = snY >> 1;

	const int snXY = snX * snY;

	// base spatial frequency

	int n_points = pc_data.n_points;
	const Real sf_base = 1.0 / (ppX * fftSegSize);

	input = new Complex<Real>*[snXY];
	for (int i = 0; i < snXY; i++) {
		input[i] = new Complex<Real>[fftsSegSize];
		memset(input[i], 0x00, sizeof(Complex<Real>) * fftsSegSize);
	}

	Complex<Real>* result = new Complex<Real>[fftsSegSize];


	for (int ch = 0; ch < context_.waveNum; ch++)
	{
		Real lambda = context_.wave_length[ch];

		for (int i = 0; i < n_points; i++)
		{
			Point pt = pc_data.vertices[i].point;
			//Real ampitude = pc_data.vertices[i].phase; // why phase?
			Real amplitude = pc_data.vertices[i].color.color[ch]; // red
			Real phase = pc_data.vertices[i].phase;
			pt.pos[_X] *= pc_config.scale[_X];
			pt.pos[_Y] *= pc_config.scale[_Y];
			pt.pos[_Z] *= pc_config.scale[_Z];
			pt.pos[_Z] -= distance;

			CalcSpatialFrequency(&pt, lambda, is_accurate);
			CalcCompensatedPhase(&pt, amplitude, phase, lambda, is_accurate);
		}
		if (is_accurate)
		{
			for (int y = 0; y < snY; y++) {

				for (int x = 0; x < snX; x++) {

					int segyy = y * snX + x;
					int segxx = coefficient_cy[y] * fftSegSize + coefficient_cx[x];

					if (segyy < snXY)
					{
						fft2(input[segyy], result, fftSegSize, fftSegSize, OPH_BACKWARD, false, false);

						for (int m = 0; m < segSize; m++)
						{
							int yy = y * segSize + m;
							int xx = x * segSize;
							int mm = m + ffthSegSize;
							for (int n = 0; n < segSize; n++)
							{
								complex_H[ch][yy * pnX + (xx + n)] += result[(m + ffthSegSize - hSegSize) * fftSegSize + (n + ffthSegSize - hSegSize)][_RE];
							}
						}
					}
				}
			}
		}
		else
		{
			for (int y = 0; y < snY; y++) {

				for (int x = 0; x < snX; x++) {

					int segyy = y * snX + x;
					int segxx = coefficient_cy[y] * segSize + coefficient_cx[x];

					if (segyy < snXY)
					{
						fft2(input[segyy], result, segSize, segSize, OPH_BACKWARD, false, false);

						for (int m = 0; m < segSize; m++)
						{
							for (int n = 0; n < segSize; n++)
							{
								int segxx = m * segSize + n;

								complex_H[ch][(y * segSize + m) * pnX + (x * segSize + n)] = result[m * segSize + n][_RE];
							}
						}
					}
				}
			}
		}
	}


	//fftFree();
	delete[] result;

	for (int i = 0; i < snXY; i++) {
		delete[] input[i];
	}
	delete[] input;	
}


void ophPAS::CalcSpatialFrequency(Point *pt, Real lambda, bool accurate)
{
	int segSize = FFT_SEGMENT_SIZE;
	int hSegSize = segSize >> 1;
	int snX = context_.pixel_number[_X] / segSize;
	int snY = context_.pixel_number[_Y] / segSize;
	Real ppX = context_.pixel_pitch[_X];
	Real ppY = context_.pixel_pitch[_Y];
	int hsnX = snX >> 1;
	int hsnY = snY >> 1;

	Real thetaX = pc_config.tilt_angle[_X];
	Real thetaY = pc_config.tilt_angle[_Y];
	Real sf_base = 1.0 / (ppX * segSize);


	if (accurate)
	{
		Real pi2 = M_PI * 2;
		Real tempX = sf_base * hSegSize * ppX;
		Real tempY = sf_base * hSegSize * ppY;

		for (int x = 0; x < snX; x++) {
			xc[x] = ((x - hsnX) * segSize + hSegSize) * ppX;
			Real theta_cx = (xc[x] - pt->pos[_X]) / pt->pos[_Z];
			Real sf_cx = (theta_cx + thetaX) / lambda;
			Real val = sf_cx >= 0 ? 0.5 : -0.5;
			int pp_cx = (int)(sf_cx / sf_base + val);
			coefficient_cx[x] = (abs(pp_cx) < hSegSize) ? ((segSize - pp_cx) % segSize) : 0;
			compensation_cx[x] = pi2 * ((xc[x] - pt->pos[_X]) * sf_cx + pp_cx * tempX);
		}

		for (int y = 0; y < snY; y++) {
			yc[y] = ((y - hsnY) * segSize + hSegSize) * ppY;
			Real theta_cy = (yc[y] - pt->pos[_Y]) / pt->pos[_Z];
			Real sf_cy = (theta_cy + thetaY) / lambda;
			Real val = sf_cy >= 0 ? 0.5 : -0.5;
			int pp_cy = (int)(sf_cy / sf_base + val);
			coefficient_cy[y] = (abs(pp_cy) < hSegSize) ? ((segSize - pp_cy) % segSize) : 0;
			compensation_cy[y] = pi2 * ((yc[y] - pt->pos[_Y]) * sf_cy + pp_cy * tempY);
		}
	}
	else
	{
		for (int x = 0; x < snX; x++) {
			xc[x] = ((x - hsnX) * segSize + hSegSize) * ppX;
			Real theta_cx = (xc[x] - pt->pos[_X]) / pt->pos[_Z];
			Real sf_cx = (theta_cx + thetaX) / lambda;
			Real val = sf_cx >= 0 ? 0.5 : -0.5;
			int pp_cx = (int)(sf_cx / sf_base + val);
			coefficient_cx[x] = (abs(pp_cx) < hSegSize) ? ((segSize - pp_cx) % segSize) : 0;
		}

		for (int y = 0; y < snY; y++) {
			yc[y] = ((y - hsnY) * segSize + hSegSize) * ppY;
			Real theta_cy = (yc[y] - pt->pos[_Y]) / pt->pos[_Z];
			Real sf_cy = (theta_cy + thetaY) / lambda;
			Real val = sf_cy >= 0 ? 0.5 : -0.5;
			int pp_cy = (int)(sf_cy / sf_base + val);
			coefficient_cy[y] = (abs(pp_cy) < hSegSize) ? ((segSize - pp_cy) % segSize) : 0;
		}
	}
}

void ophPAS::CalcCompensatedPhase(Point *pt, Real amplitude, Real phase, Real lambda, bool accurate)
{
	int segSize = FFT_SEGMENT_SIZE;
	int hSegSize = segSize >> 1;
	int snX = context_.pixel_number[_X] / segSize;
	int snY = context_.pixel_number[_Y] / segSize;
	Real ppX = context_.pixel_pitch[_X];
	Real ppY = context_.pixel_pitch[_Y];
	int hsnX = snX >> 1;
	int hsnY = snY >> 1;
	Real zz = pt->pos[_Z] * pt->pos[_Z];
	Real pi2 = M_PI * 2;

	if (accurate)
	{
		for (int y = 0; y < snY; y++)
		{
			Real yy = (yc[y] - pt->pos[_Y]) * (yc[y] - pt->pos[_Y]);
			for (int x = 0; x < snX; x++)
			{
				int segyy = y * snX + x;
				int segxx = coefficient_cy[y] * segSize + coefficient_cx[x];
				Real xx = (xc[x] - pt->pos[_X]) * (xc[x] - pt->pos[_X]);
				Real r = sqrt(xx + yy + zz);

				Real theta = lambda * r + phase + compensation_cy[y] + compensation_cx[x];
				Real theta_c = theta;
				Real theta_s = theta + M_PI;

				int idx_c = ((int)(theta_c * NUMTBL / pi2)) & NUMTBL2;
				int idx_s = ((int)(theta_s * NUMTBL / pi2)) & NUMTBL2;
				input[segyy][segxx][_RE] += amplitude * LUTCos[idx_c];
				input[segyy][segxx][_IM] += amplitude * LUTSin[idx_s];
			}
		}
	}
	else
	{
		for (int y = 0; y < snY; y++)
		{
			Real yy = (yc[y] - pt->pos[_Y]) * (yc[y] - pt->pos[_Y]);
			for (int x = 0; x < snX; x++)
			{
				int segyy = y * snX + x;
				int segxx = coefficient_cy[y] * segSize + coefficient_cx[x];
				Real xx = (xc[x] - pt->pos[_X]) * (xc[x] - pt->pos[_X]);
				Real r = sqrt(xx + yy + zz);

				Real theta = lambda * r;
				Real theta_c = theta;
				Real theta_s = theta + M_PI;

				int idx_c = ((int)(theta_c * NUMTBL / pi2)) & NUMTBL2;
				int idx_s = ((int)(theta_s * NUMTBL / pi2)) & NUMTBL2;
				input[segyy][segxx][_RE] += amplitude * LUTCos[idx_c];
				input[segyy][segxx][_IM] += amplitude * LUTSin[idx_s];
			}
		}
	}
}

void ophPAS::encodeHologram(const vec2 band_limit, const vec2 spectrum_shift)
{
	if (complex_H == nullptr) {
		LOG("Not found diffracted data.");
		return;
	}


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

	for (uint i = 0; i < pnX; i++)
		x_o[i] = (-ss[_X] / 2) + (ppX * i) + (ppX / 2);

	for (uint i = 0; i < pnY; i++)
		y_o[i] = (ss[_Y] - ppY) - (ppY * i);

	Real* xx_o = new Real[pnXY];
	Real* yy_o = new Real[pnXY];

	for (int i = 0; i < pnXY; i++)
		xx_o[i] = x_o[i % pnX];


	for (uint i = 0; i < pnX; i++)
		for (uint j = 0; j < pnY; j++)
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

}

void ophPAS::encoding(unsigned int ENCODE_FLAG)
{
	ophGen::encoding(ENCODE_FLAG);
}
