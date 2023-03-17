#include "ophIFTA.h"
#include "sys.h"
#include <omp.h>
#include <stdlib.h>
#include "include.h"
#include "tinyxml2.h"

ophIFTA::ophIFTA()
{
	imgRGB = nullptr;
	imgDepth = nullptr;
	imgOutput = nullptr;
	width = 0;
	height = 0;
	bytesperpixel = 0;
	nearDepth = 0.0;
	farDepth = 0.0;
	nDepth = 0;
	nIteration = 0;
}


ophIFTA::~ophIFTA()
{
	if (imgOutput) delete[] imgOutput;
	if (imgRGB) delete[] imgRGB;
	if (imgDepth) delete[] imgDepth;
}

using namespace oph;
Real ophIFTA::generateHologram()
{
	if ((!imgRGB && m_config.num_of_depth == 1) || (!imgRGB && !imgDepth))
		return 0.0;

	srand(time(NULL));

	auto begin = CUR_TIME;

	const uint nWave = context_.waveNum;
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const long long int pnXY = pnX * pnY;
	const Real ssX = context_.ss[_X];
	const Real ssY = context_.ss[_Y];
	const Real nDepth = m_config.num_of_depth;
	Real nearDepth = m_config.near_depthmap;
	Real farDepth = m_config.far_depthmap;
	int nIteration = m_config.num_of_iteration;
	int num_thread = 1;

	Real *waveRatio = new Real[nWave];
	for (uint i = 0; i < nWave; i++) {
		waveRatio[i] = context_.wave_length[nWave - 1] / context_.wave_length[i];
	}

	// constants
	Real d = (nDepth == 1) ? 0.0 : (farDepth - nearDepth) / (nDepth - 1);
	Real z = farDepth;
	Real pi2 = 2 * M_PI;
	uchar m = getMax(imgDepth, pnX, pnY);
	Real *depth_quant = new Real[pnXY];
	memset(depth_quant, 0, sizeof(Real) * pnXY);

	for (int depth = nDepth; depth > 0; depth--) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < pnXY; i++) {
			depth_quant[i] += (imgDepth[i] > (depth*m / nDepth)) ? 1 : 0;
		}
	}
	if (imgOutput) {
		delete[] imgOutput;
		imgOutput = nullptr;
	}
	imgOutput = new uchar[pnXY * bytesperpixel];
	for (int depth = 0; depth < nDepth; depth++) {
		z = farDepth - (d*depth);

		for (uint ch = 0; ch < nWave; ch++) {
			uchar *img = new uchar[pnXY];
			separateColor(ch, pnX, pnY, imgRGB, img);
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < pnXY; i++) {
				Real val = (depth_quant[i] == depth) ? 1.0 : 0.0;
				img[i] = val * (Real)img[i];
			}
			Real lambda = context_.wave_length[ch];
			Real k = 2 * M_PI / lambda;
			Real hssX = lambda * z / ppX;
			Real hssY = lambda * z / ppY;
			Real hppX = hssX / pnX;
			Real hppY = hssY / pnY;

			Real hStartX = -hssX / 2;
			Real hStartY = -hssY / 2;
			Real startX = -ssX / 2;
			Real startY = -ssY / 2;

			Complex<Real> c1(0.0, lambda * z);
			Complex<Real> c2(0.0, -k / (2 * lambda));
			Complex<Real> c3(0.0, -k / (2 * z));
			uchar *img_tmp = nullptr;
			uchar *imgScaled = nullptr;

			// blue에서는 리사이즈 할 필요가 없다.
			if (ch < 2)
			{
				int scaleX = round(pnX * 4 * waveRatio[ch]);
				int scaleY = round(pnY * 4 * waveRatio[ch]);

				int ww = pnX * 4;
				int hh = pnY * 4;
				img_tmp = new uchar[ww * hh];
				imgScaled = new uchar[scaleX * scaleY];
				imgScaleBilinear(img, imgScaled, pnX, pnY, scaleX, scaleY);
				delete[] img;
				Real ppY2 = 2 * ppY;
				Real ppX2 = 2 * ppX;

				memset(img_tmp, 0, sizeof(uchar) * ww * hh);

				int h1 = round((hh - scaleY) / 2);
				int w1 = round((ww - scaleX) / 2);

				// 이미지를 중앙으로 조정
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int y = 0; y < scaleY; y++) {
					for (int x = 0; x < scaleX; x++) {
						img_tmp[(y + h1)*ww + x + w1] = imgScaled[y*scaleX + x];
					}
				}
				img = new uchar[pnXY];
				imgScaleBilinear(img_tmp, img, ww, hh, pnX, pnY);
			}
			Real *target = new Real[pnXY];
			Complex<Real> *result1 = new Complex<Real>[pnXY];
			Complex<Real> *result2 = new Complex<Real>[pnXY];
			Complex<Real> *kernel = new Complex<Real>[pnXY];
			Complex<Real> *kernel2 = new Complex<Real>[pnXY];
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int y = 0; y < pnY; y++) {
				int offset = y * pnX;

				Real ty = hStartY + (y * hppY);
				Real yy = startY + (y * ppY);

				for (int x = 0; x < pnX; x++) {
					target[offset + x] = (Real)img[offset + x];
					Real ran;
#ifdef ANOTHER_RAND
					ran = rand(0.0, 1.0);
#else
					ran = ((Real)rand() / RAND_MAX) * 1.0;
#endif
					Complex<Real> tmp, c4;
					if (ran < 1.0) {
						tmp(0.0, ran * 2 * M_PI);
						c4 = tmp.exp();
					}
					else
						c4(1.0, 0.0);

					Real tx = hStartX + (x * hppX);
					Real txy = tx * tx + ty * ty;
					Real xx = startX + (x * ppX);
					Real xy = xx * xx + yy * yy;

					Complex<Real> t = (c2 * txy).exp();
					kernel[offset + x] = c1 * t;
					kernel2[offset + x] = (c3 * xy).exp();
					result1[offset + x] = target[offset + x] * c4;
					result1[offset + x] *= kernel[offset + x];
				}
			}
			Complex<Real> *tmp = new Complex<Real>[pnXY];
			Complex<Real> *tmp2 = new Complex<Real>[pnXY];
			memset(tmp, 0, sizeof(Complex<Real>) * pnXY);
			memset(tmp2, 0, sizeof(Complex<Real>) * pnXY);
			fftShift(pnX, pnY, result1, tmp);
			fft2(ivec2(pnX, pnY), tmp, OPH_FORWARD, OPH_ESTIMATE);
			fftExecute(tmp, true);
			memset(result1, 0, sizeof(Complex<Real>) * pnXY);

			if (nIteration == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int j = 0; j < pnXY; j++) {
					result2[j] = tmp[j] / kernel2[j];
					complex_H[ch][j] += result2[j];
				}
			}
			else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int y = 0; y < pnXY; y++) {
					result2[y] = tmp[y] / kernel2[y];
				}

				memset(tmp, 0, sizeof(Complex<Real>) * pnXY);

				for (int i = 0; i < nIteration; i++) {
					for (int j = 0; j < pnXY; j++) {
						result2[j] = tmp[j] / kernel2[j];
						tmp[j][_RE] = 0.0;
						tmp[j][_IM] = result2[j].angle();
						result1[j] = tmp[j].exp() * kernel2[j];
					}

					// FFT
					memset(tmp, 0, sizeof(Complex<Real>) * pnXY);
					fft2(ivec2(pnX, pnY), result1, OPH_FORWARD, OPH_ESTIMATE);
					fftExecute(tmp);
					fftShift(pnX, pnY, tmp, result1);

					for (int j = 0; j < pnXY; j++) {
						result1[j] /= kernel[j];
						Complex<Real> aa(0.0, result1[j].angle());
						aa = aa.exp();
						result2[j] = (target[j] / 255.0) * aa;
						result2[j] *= kernel[j];
					}
					fftShift(pnX, pnY, result2, tmp);
					fft2(ivec2(pnX, pnY), tmp, OPH_FORWARD, OPH_ESTIMATE);
					fftExecute(tmp2, true);

					for (int j = 0; j < pnXY; j++) {
						result2[j] = tmp2[j] / kernel2[j];
					}
					LOG("Iteration (%d / %d)\n", i + 1, nIteration);
				}
				memset(img, 0, pnXY);
				for (int j = 0; j < pnXY; j++) {
					complex_H[ch][j] += result2[j];
				}
			}
			m_nProgress = (int)((Real)(depth*nWave + ch) * 100 / ((Real)nDepth * nWave));

			delete[] kernel;
			delete[] kernel2;
			delete[] tmp;
			delete[] tmp2;
			delete[] result2;
			delete[] result1;
			delete[] target;
			delete[] img_tmp;
			delete[] imgScaled;
			delete[] img;

			LOG("Color Channel (%d / %d) %lf(s)\n", ch + 1, nWave, ELAPSED_TIME(begin, CUR_TIME));
		}
		LOG("Depth Level (%d / %d) %lf(s)\n", depth + 1, (int)nDepth, ELAPSED_TIME(begin, CUR_TIME));
	}
	delete[] depth_quant;
	delete[] waveRatio;
	auto end = CUR_TIME;
	LOG("\nTotal Elapsed Time : %lf(s)\n\n", ELAPSED_TIME(begin, end));
	return  ELAPSED_TIME(begin, end);
}

bool ophIFTA::normalize()
{
	const uint nWave = context_.waveNum;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const long long int pnXY = pnX * pnY;
	const Real pi2 = M_PI * 2;

	for (uint ch = 0; ch < nWave; ch++) {
		for (int i = 0; i < pnXY; i++) {
			m_lpNormalized[ch][i] = m_lpEncoded[ch][i] / pi2 * 255;
		}
	}
	return true;
}

void ophIFTA::encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND)
{
	auto begin = CUR_TIME;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;
	Complex<Real>* dst = new Complex<Real>[pnX * pnY];

	for (uint ch = 0; ch < nChannel; ch++) {
		fft2(context_.pixel_number, complex_H[ch], OPH_BACKWARD);
		fft2(complex_H[ch], dst, pnX, pnY, OPH_BACKWARD);

		if (ENCODE_FLAG == ophGen::ENCODE_SSB) {
			ivec2 location;
			switch (SSB_PASSBAND) {
			case SSB_TOP:
				location = ivec2(0, 1);
				break;
			case SSB_BOTTOM:
				location = ivec2(0, -1);
				break;
			case SSB_LEFT:
				location = ivec2(-1, 0);
				break;
			case SSB_RIGHT:
				location = ivec2(1, 0);
				break;
			}

			encodeSideBand(true, location);
		}
		else ophGen::encoding(ENCODE_FLAG, SSB_PASSBAND, dst);
	}
	delete[] dst;
	auto end = CUR_TIME;
	LOG("Elapsed Time: %lf(s)\n", ELAPSED_TIME(begin, end));
}

bool ophIFTA::readConfig(const char* fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	using namespace tinyxml2;
	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	if (xml_doc.LoadFile(fname) != XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

	// about viewing window
	auto next = xml_node->FirstChildElement("DepthLevel");
	if (!next || XML_SUCCESS != next->QueryIntText(&m_config.num_of_depth))
		m_config.num_of_depth = 1;
	next = xml_node->FirstChildElement("NearOfDepth");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&m_config.near_depthmap))
		return false;
	next = xml_node->FirstChildElement("FarOfDepth");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&m_config.far_depthmap))
		return false;
	next = xml_node->FirstChildElement("NumOfIteration");
	if (!next || XML_SUCCESS != next->QueryIntText(&m_config.num_of_iteration))
		m_config.num_of_iteration = 1;

	initialize();
	return true;
}


bool ophIFTA::readImage(const char* fname, bool bRGB)
{
	bool ret = getImgSize(width, height, bytesperpixel, fname);

	if (ret) {
		const uint pnX = context_.pixel_number[_X];
		const uint pnY = context_.pixel_number[_Y];
		uchar *imgIFTA = nullptr;

		if (imgIFTA != nullptr) {
			delete[] imgIFTA;
			imgIFTA = nullptr;
		}
		uchar *imgTmp = loadAsImg(fname);
		imgIFTA = new uchar[pnX * pnY * bytesperpixel];
		memset(imgIFTA, 0, pnX * pnY * bytesperpixel);
		imgScaleBilinear(imgTmp, imgIFTA, width, height, pnX, pnY, bytesperpixel);

		delete[] imgTmp;

		if (bRGB) imgRGB = imgIFTA;
		else imgDepth = imgIFTA;
	}
	return ret;
}

uchar ophIFTA::getMax(uchar *src, int width, int height)
{
	const long long int size = width * height;
	uchar max = 0;

	for (int i = 0; i < size; i++) {
		if (*(src + i) > max) max = *(src + i);
	}
	return max;
}