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

ophACPAS::ophACPAS()
	: ophGen()
	, m_pHologram(nullptr)
	, n_points(-1)
	
{
}

ophACPAS::~ophACPAS()
{
	delete[] m_pHologram;
}


bool ophACPAS::readConfig(const char* fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	bool bRet = true;
	/*XML parsing*/

	using namespace tinyxml2;
	tinyxml2::XMLDocument xml_doc;
	XMLNode* xml_node = nullptr;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("<FAILED> Wrong file ext.\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fname);
	if (ret != XML_SUCCESS)
	{
		LOG("<FAILED> Loading file.\n");
		return false;
	}

	xml_node = xml_doc.FirstChild();

	char szNodeName[32] = { 0, };
	sprintf(szNodeName, "ScaleX");
	// about point
	auto next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.scale[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "ScaleY");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.scale[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "ScaleZ");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.scale[_Z]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Distance");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.distance))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	// ACPAS Environment
	sprintf(szNodeName, "CGH Width");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&env.CghWidth))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "CGH Height");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&env.CghHeight))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "CGH Scale");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryFloatText(&env.CGHScale))
	{
		LOG("<FAILED> Not found node : \'%s\' (Float) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Segmentation Size");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&env.SegmentationSize))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "FFT Segmentation Size");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&env.fftSegmentationSize))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Red WaveLength");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryFloatText(&env.rWaveLength))
	{
		LOG("<FAILED> Not found node : \'%s\' (Float) \n", szNodeName);
		bRet = false;
	}

	sprintf(szNodeName, "Tilting angle on x axis");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryFloatText(&env.ThetaX))
	{
		LOG("<FAILED> Not found node : \'%s\' (Float) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Tilting angle on y axis");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryFloatText(&env.ThetaY))
	{
		LOG("<FAILED> Not found node : \'%s\' (Float) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Default depth");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryFloatText(&env.DefaultDepth))
	{
		LOG("<FAILED> Not found node : \'%s\' (Float) \n", szNodeName);
		bRet = false;
	}

	sprintf(szNodeName, "3D point interval on x axis");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryFloatText(&env.xInterval))
	{
		LOG("<FAILED> Not found node : \'%s\' (Float) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "3D point interval on y axis");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryFloatText(&env.yInterval))
	{
		LOG("<FAILED> Not found node : \'%s\' (Float) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Hologram interval on xi axis");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryFloatText(&env.xiInterval))
	{
		LOG("<FAILED> Not found node : \'%s\' (Float) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Hologram interval on eta axis");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryFloatText(&env.etaInterval))
	{
		LOG("<FAILED> Not found node : \'%s\' (Float) \n", szNodeName);
		bRet = false;
	}

	initialize();

	context_.k = (2 * M_PI) / context_.wave_length[0];
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	Openholo::setPixelNumberOHC(context_.pixel_number);
	Openholo::setPixelPitchOHC(context_.pixel_pitch);
	Openholo::setWavelengthOHC(context_.wave_length[0], LenUnit::m);

	LOG("**************************************************\n");
	LOG("   Read Config (Accurate Phase-Added Stereogram)  \n");
	LOG("1) Focal Length : %.5lf\n", pc_config_.distance);
	LOG("2) Object Scale : %.5lf / %.5lf / %.5lf\n", pc_config_.scale[_X], pc_config_.scale[_Y], pc_config_.scale[_Z]);
	LOG("**************************************************\n");

	return bRet;
}

int ophACPAS::loadPointCloud(const char* pc_file)
{
	n_points = ophGen::loadPointCloud(pc_file, &pc_data_);
	return n_points;
}

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


void ophACPAS::DataInit()
{
	if (m_pHologram == nullptr)
		m_pHologram = new double[getContext().pixel_number[_X] * getContext().pixel_number[_Y]];
	memset(m_pHologram, 0x00, sizeof(double) * getContext().pixel_number[_X] * getContext().pixel_number[_Y]);

	for (int i = 0; i < NUMTBL; i++) {
		float theta = (float)M2_PI * (float)(i + i - 1) / (float)(2 * NUMTBL);
		m_COStbl[i] = (float)cos(theta);
		m_SINtbl[i] = (float)sin(theta);
	}
}

int ophACPAS::ACPASCalcuation(unsigned char * cghfringe)
{
	double Max = -1E9, Min = 1E9;
	int cghwidth = getContext().pixel_number[_X];
	int cghheight = getContext().pixel_number[_Y];

	DataInit();
	ACPAS();

	for (int i = 0; i < cghheight; i++) {
		for (int j = 0; j < cghwidth; j++) {
			if (Max < m_pHologram[i*cghwidth + j])	Max = m_pHologram[i*cghwidth + j];
			if (Min > m_pHologram[i*cghwidth + j])	Min = m_pHologram[i*cghwidth + j];
		}
	}

	for (int i = 0; i < cghheight; i++) {
		for (int j = 0; j < cghwidth; j++) {
			double temp = 1.0*(((m_pHologram[i*cghwidth + j] - Min) / (Max - Min))*255. + 0.5);
			if (temp >= 255.0)  cghfringe[i*cghwidth + j] = 255;
			else				cghfringe[i*cghwidth + j] = (unsigned char)(temp);
		}
	}

	return 0;
}

void ophACPAS::ACPAS()
{
	float xiInterval = env.xiInterval;
	float etaInterval = env.etaInterval;
	float rLamda = env.rWaveLength;
	float rWaveNum = env.rWaveNumber;
	float thetaX = env.ThetaX;
	float thetaY = env.ThetaY;

	int hsegSize = env.SegmentationSize >> 1;
	int dsegSize = env.SegmentationSize * env.SegmentationSize;
	int segNumx = (int)(env.CghWidth / env.SegmentationSize);
	int segNumy = (int)(env.CghHeight/ env.SegmentationSize);
	int hsegNumx = segNumx >> 1;
	int hsegNumy = segNumy >> 1;

	int FFThsegSize = env.fftSegmentationSize >> 1;
	int FFTdsegSize = env.fftSegmentationSize * env.fftSegmentationSize;

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
	float	sf_base = 1.0 / (env.xiInterval * env.fftSegmentationSize);
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
	for (int i = 0; i < segNumy; i++) {
		for (int j = 0; j < segNumx; j++) {
			inRe[i * segNumx + j] = new double[FFTdsegSize];
			inIm[i * segNumx + j] = new double[FFTdsegSize];
			memset(inRe[i*segNumx + j], 0x00, sizeof(double) * FFTdsegSize);
			memset(inIm[i*segNumx + j], 0x00, sizeof(double) * FFTdsegSize);
		}
	}

	in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FFTdsegSize);
	out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FFTdsegSize);
	memset(in, 0x00, sizeof(fftw_complex) * FFTdsegSize);
	memset(m_pHologram, 0x00, sizeof(double) * env.CghWidth * env.CghHeight);

	for (int i = 0; i < segNumy; i++)
		yc[i] = ((i - hsegNumy) * env.SegmentationSize + hsegSize) * env.etaInterval;
	for (int i = 0; i < segNumx; i++)
		xc[i] = (((i- hsegNumx) * env.SegmentationSize) + hsegSize) * env.xiInterval;

	for (int i = 0; i < pc_data_.n_points; i++)
	{
		float x = pc_data_.vertices[i].point.pos[_X] * pc_config_.scale[_X];
		float y = pc_data_.vertices[i].point.pos[_Y] * pc_config_.scale[_Y];
		float z = pc_data_.vertices[i].point.pos[_Z] * pc_config_.scale[_Z] - pc_config_.distance;
		float amplitude = pc_data_.vertices[i].color.color[_R]; // 1 channel
		float phase = pc_data_.vertices[i].phase;

		for (int j = 0; j < segNumy; j++)
		{
			float theta_cy = (yc[j] - y) - z;
			SFrequency_cy[j] = (theta_cy + thetaY) / env.rWaveLength;
			PickPoint_cy[j] = (SFrequency_cy[j] >= 0 ) ? (int)(SFrequency_cy[j] / sf_base + 0.5) : (int)(SFrequency_cy[j] / sf_base - 0.5);
			Coefficient_cy[j] = (abs(PickPoint_cy[j]) < FFThsegSize) ? ((env.fftSegmentationSize - PickPoint_cy[j]) % env.fftSegmentationSize) : 0;
			Compensation_cy[j] = (float)(2 * PI * ((yc[j] - y) * SFrequency_cy[j] + PickPoint_cy[j] * sf_base * FFThsegSize * xiInterval));
		}
		for (int j = 0; j < segNumx; j++)
		{
			float theta_cx = (xc[j] - x) / z;
			SFrequency_cx[j] = (theta_cx + thetaX) / env.rWaveLength;
			PickPoint_cx[j] = (SFrequency_cx[j] >= 0) ? (int)(SFrequency_cx[j] / sf_base + 0.5) : (int)(SFrequency_cx[j] / sf_base - 0.5);
			Coefficient_cx[j] = (abs(PickPoint_cx[j]) < FFThsegSize) ? ((env.fftSegmentationSize - PickPoint_cx[j]) % env.fftSegmentationSize) : 0;
			Compensation_cx[j] = (float)(2 * PI * ((xc[j] - x) * SFrequency_cx[j] + PickPoint_cx[j] * sf_base * FFThsegSize * etaInterval));
		}
		for (int j = 0; j < segNumy; j++) {
			for (int k = 0; k < segNumx; k++) {
				int idx = j* segNumx + k;
				int idx2 = Coefficient_cy[j] * env.fftSegmentationSize + Coefficient_cx[k];
				float R = sqrt((xc[k] - x) * (xc[k] - x) + (yc[j] - y) * (yc[j] - y) + z * z);
				theta = rWaveNum * R + phase + Compensation_cy[j] + Compensation_cx[k];
				//+ dPhaseSFy[j] + dPhaseSFx[k];
				theta_c = theta;
				theta_s = theta + PI;
				dtheta_c = ((int)(theta_c * NUMTBL / M2_PI));
				dtheta_s = ((int)(theta_s * NUMTBL / M2_PI));
				idx_c = (dtheta_c) & (NUMTBL2);
				idx_s = (dtheta_s) & (NUMTBL2);
				inRe[idx][idx2] += (double)(amplitude * m_COStbl[idx_c]);
				inIm[idx][idx2] += (double)(amplitude * m_SINtbl[idx_s]);
			}
		}
	}

	plan = fftw_plan_dft_2d(env.SegmentationSize, env.SegmentationSize, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

	for (int j = 0; j < segNumy; j++) {
		for (int k = 0; k < segNumx; k++) {
			int idx = j * segNumx + k;
			memset(in, 0x00, sizeof(fftw_complex) * FFTdsegSize);
			for (int l = 0; l < env.fftSegmentationSize; l++) {
				for (int m = 0; m < env.fftSegmentationSize; m++) {
					int idx2 = l * env.fftSegmentationSize + m;
					in[l * env.fftSegmentationSize + m][0] = inRe[idx][idx2];
					in[l * env.fftSegmentationSize + m][1] = inIm[idx][idx2];
				}
			}
			fftw_execute(plan);
			for (int l = 0; l < env.SegmentationSize; l++) {
				for (int m = 0; m < env.SegmentationSize; m++) {
					m_pHologram[(j * env.SegmentationSize + l) * env.CghWidth + (k * env.SegmentationSize + m)] +=
						out[(l + FFThsegSize - hsegSize) * env.fftSegmentationSize + (m + FFThsegSize - hsegSize)][0];// - out[l * SEGSIZE + m][1];
				}
			}
		}
	}

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
	for (int i = 0; i < segNumy; i++) {
		for (int j = 0; j < segNumx; j++) {
			delete[] inRe[i * segNumx + j];
			delete[] inIm[i * segNumx + j];
		}
	}
	delete[] inRe;
	delete[] inIm;
}
