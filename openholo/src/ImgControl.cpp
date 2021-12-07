#include "ImgControl.h"
#include <string.h>
#include <gdiplus.h>
#include <Shlwapi.h>
#include <omp.h>
#include "sys.h"
#include "function.h"
#pragma comment(lib, "gdiplus.lib")

ImgControl* ImgControl::instance = nullptr;

ImgControl::ImgControl()
{
}


ImgControl::~ImgControl()
{
}

using namespace Gdiplus;
bool ImgControl::Save(const char *path, BYTE *pSrc, UINT len, int quality)
{
	Status stat = Ok;
	GdiplusStartupInput gsi;
	ULONG_PTR token = NULL;

	if (GdiplusStartup(&token, &gsi, NULL) == Ok) {
		BYTE *pDest = new BYTE[len];
		memcpy(pDest, pSrc, len);

		CComPtr<IStream> pStream;
		pStream.Attach(SHCreateMemStream(pDest, len));
		Image img(pStream, FALSE);

		CLSID clsid;
		wchar_t format[256] = { 0, };
		wchar_t wpath[256] = { 0, };	

		int len = MultiByteToWideChar(CP_ACP, 0, (LPCTSTR)path, (int)strlen(path), NULL, NULL);
		MultiByteToWideChar(CP_ACP, 0, (LPCTSTR)path, (int)strlen(path), wpath, len);

		wsprintfW(format, L"image/%s", PathFindExtensionW(wpath) + 1);

		bool bHasParam = false;
		EncoderParameters params;
		memset(&params, 0, sizeof(EncoderParameters));
		if (!_stricmp(PathFindExtensionA(path) + 1, "jpg") ||
			!_stricmp(PathFindExtensionA(path) + 1, "jpeg")) {

			wsprintfW(format, L"image/%s", L"jpeg");

			ULONG q = (quality > 100 || quality < 0) ? 100 : quality;
			params.Count = 1;
			params.Parameter[0].Guid = EncoderQuality;
			params.Parameter[0].Type = EncoderParameterValueTypeLong;
			params.Parameter[0].NumberOfValues = 1;
			params.Parameter[0].Value = &q;
			bHasParam = true;
		}
		else if (!_stricmp(PathFindExtensionA(path) + 1, "tif")) {
			wsprintfW(format, L"image/%s", L"tiff");
		}
		if (GetEncoderClsid(format, &clsid) != -1) {
			stat = img.Save(wpath, &clsid, bHasParam ? &params : NULL);
		}
		pStream.Detach();
		pStream.Release();
		delete[] pDest;
	}
 	GdiplusShutdown(token);
	return stat == Ok ? true : false;
}

int ImgControl::GetEncoderClsid(const WCHAR *format, CLSID *pClsid)
{
	UINT nEncoder = 0; // number of image encoders
	UINT nSize = 0; // size of the image encoder array in bytes

	ImageCodecInfo* pImageCodecInfo = NULL;

	GetImageEncodersSize(&nEncoder, &nSize);
	if (nSize == 0)
		return -1;  // Failure

	pImageCodecInfo = (ImageCodecInfo*)malloc(nSize);
	if (pImageCodecInfo == NULL)
		return -1;  // Failure

	GetImageEncoders(nEncoder, nSize, pImageCodecInfo);

	for (UINT j = 0; j < nEncoder; ++j)
	{
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;                                         // Success
		}
	}
	free(pImageCodecInfo);
	return -1;
}


void ImgControl::Resize(unsigned char* src, unsigned char* dst, int w, int h, int neww, int newh, int ch)
{
	if (src == nullptr || dst == nullptr) return;
	auto begin = CUR_TIME;
	int nBytePerLine = ((w * ch) + 3) & ~3; // src
	int nNewBytePerLine = ((neww * ch) + 3) & ~3; // dst
	int y;
#ifdef _OPENMP
#pragma omp parallel for private(y) firstprivate(nNewBytePerLine, nBytePerLine)
#endif
	for (y = 0; y < newh; y++)
	{
		int nbppY = y * nNewBytePerLine;
		for (int x = 0; x < neww; x++)
		{
			float gx = (x / (float)neww) * (w - 1);
			float gy = (y / (float)newh) * (h - 1);

			int gxi = (int)gx;
			int gyi = (int)gy;

			if (ch == 1) {
				uint32_t a00, a01, a10, a11;

				a00 = src[gxi + 0 + gyi * nBytePerLine];
				a01 = src[gxi + 1 + gyi * nBytePerLine];
				a10 = src[gxi + 0 + (gyi + 1) * nBytePerLine];
				a11 = src[gxi + 1 + (gyi + 1) * nBytePerLine];

				float dx = gx - gxi;
				float dy = gy - gyi;

				float w1 = (1 - dx) * (1 - dy);
				float w2 = dx * (1 - dy);
				float w3 = (1 - dx) * dy;
				float w4 = dx * dy;

				dst[x + y * neww] = int(a00 * w1 + a01 * w2 + a10 * w3 + a11 * w4);
			}
			else if (ch == 3) {
				uint32_t b00[3], b01[3], b10[3], b11[3];
				int srcX = gxi * ch;
				int dstX = x * ch;

				b00[0] = src[srcX + 0 + gyi * nBytePerLine];
				b00[1] = src[srcX + 1 + gyi * nBytePerLine];
				b00[2] = src[srcX + 2 + gyi * nBytePerLine];

				b01[0] = src[srcX + 3 + gyi * nBytePerLine];
				b01[1] = src[srcX + 4 + gyi * nBytePerLine];
				b01[2] = src[srcX + 5 + gyi * nBytePerLine];

				b10[0] = src[srcX + 0 + (gyi + 1) * nBytePerLine];
				b10[1] = src[srcX + 1 + (gyi + 1) * nBytePerLine];
				b10[2] = src[srcX + 2 + (gyi + 1) * nBytePerLine];

				b11[0] = src[srcX + 3 + (gyi + 1) * nBytePerLine];
				b11[1] = src[srcX + 4 + (gyi + 1) * nBytePerLine];
				b11[2] = src[srcX + 5 + (gyi + 1) * nBytePerLine];

				float dx = gx - gxi;
				float dy = gy - gyi;

				float w1 = (1 - dx) * (1 - dy);
				float w2 = dx * (1 - dy);
				float w3 = (1 - dx) * dy;
				float w4 = dx * dy;

				dst[dstX + 0 + nbppY] = int(b00[0] * w1 + b01[0] * w2 + b10[0] * w3 + b11[0] * w4);
				dst[dstX + 1 + nbppY] = int(b00[1] * w1 + b01[1] * w2 + b10[1] * w3 + b11[1] * w4);
				dst[dstX + 2 + nbppY] = int(b00[2] * w1 + b01[2] * w2 + b10[2] * w3 + b11[2] * w4);
			}
		}
	}
	auto end = CUR_TIME;
	LOG("Image Size Scaled : (%d bit) (%dx%d) => (%dx%d) : %lf(s)\n",
		ch * 8,
		w, h, neww, newh,
		ELAPSED_TIME(begin, end));
}

bool ImgControl::Rotate(double rotate, unsigned char *src, unsigned char *dst, int w, int h, int neww, int newh, int ch)
{
	if (src == nullptr || dst == nullptr) return false;
	if (ch > 4) return false;
	auto begin = CUR_TIME;

	bool bChangeSize = false;
	if (neww != w || newh != h) {
		bChangeSize = true;
	}
	
	int nImgSize = bChangeSize ? CalcBitmapSize(neww, newh, ch) : CalcBitmapSize(w, h, ch);
	unsigned char *temp = new unsigned char[nImgSize]; //src
	unsigned char *temp2 = new unsigned char[nImgSize]; // dst

	if (bChangeSize) {
		Resize(src, temp, w, h, neww, newh, ch);
		w = neww;
		h = newh;
	}
	else {
		memcpy(temp, src, nImgSize);
	}

	int nBytePerLine = ((w * ch) + 3) & ~3;
	double radian = RADIAN(rotate);
	double cc = cos(radian);
	double ss = sin(-radian);
	double centerX = (double)w / 2.0;
	double centerY = (double)h / 2.0;
	int y;
#ifdef _OPENMP
#pragma omp parallel for private(y) firstprivate(nBytePerLine, ss, cc, centerX, centerY)
	for (y = 0; y < h; y++) {
#endif
		int dstY = y * nBytePerLine;
		for (int x = 0; x < w; x++) {
			int origX = (int)(centerX + ((double)y - centerY)*ss + ((double)x - centerX)*cc);
			int origY = (int)(centerY + ((double)y - centerY)*cc - ((double)x - centerX)*ss);

			unsigned char pixels[4] = { 0, };
			if ((origY >= 0 && origY < h) && (origX >= 0 && origX < w)) {
				int offsetX = origX * ch;
				int offsetY = origY * nBytePerLine;

				memcpy(pixels, &temp[offsetY + offsetX], sizeof(unsigned char) * ch);
			}
			memcpy(&temp2[dstY + (x * ch)], pixels, sizeof(unsigned char) * ch);
		}
	}

	memcpy(dst, temp2, nImgSize);
	delete[] temp;
	delete[] temp2;
	auto end = CUR_TIME;
	LOG("Image Rotate : (%d bit) (%dx%d) (%lf degree) : %lf(s)\n",
		ch * 8,
		w, h, rotate,
		ELAPSED_TIME(begin, end));
	return true;
}

bool ImgControl::Flip(FLIP mode, unsigned char *src, unsigned char *dst, int w, int h, int ch)
{
	if (src == nullptr || dst == nullptr) return false;
	auto begin = CUR_TIME;
	bool bOK = true;
	int nImgSize = CalcBitmapSize(w, h, ch);
	unsigned char *temp = new unsigned char[nImgSize];
	int nBytePerLine = ((w * ch) + 3) & ~3;
	if (mode == FLIP::VERTICAL) {
		int y;
#ifdef _OPENMP
#pragma omp parallel for private(y) firstprivate(nBytePerLine)
#endif
		for (y = 0; y < h; y++) {
			int offset = y * nBytePerLine;
			int offset2 = (h - y - 1) * nBytePerLine;
			for (int x = 0; x < w; x++) {
				memcpy(&temp[offset + (x * ch)], &src[offset2 + (x * ch)], sizeof(unsigned char) * ch);
			}
		}
	}
	else if (mode == FLIP::HORIZONTAL) {
		int y;
#ifdef _OPENMP
#pragma omp parallel for private(y) firstprivate(nBytePerLine)
#endif
		for (y = 0; y < h; y++) {
			int offset = y * nBytePerLine;
			for (int x = 0; x < w; x++) {
				memcpy(&temp[offset + (x * ch)], &src[offset + ((w * ch) - ((x + 1) * ch))], sizeof(unsigned char) * ch);
			}
		}
	}
	else if (mode == FLIP::BOTH) {
		int y;
#ifdef _OPENMP
#pragma omp parallel for private(y) firstprivate(nBytePerLine)
#endif
		for (y = 0; y < h; y++) {
			int offset = y * nBytePerLine;
			int offset2 = (h - y - 1) * nBytePerLine;
			for (int x = 0; x < w; x++) {
				memcpy(&temp[offset + (x * ch)], &src[offset2 + ((w * ch) - ((x + 1) * ch))], sizeof(unsigned char) * ch);
			}
		}
	}
	else {
		// do nothing
		bOK = false;
	}

	memcpy(dst, temp, sizeof(unsigned char) * nImgSize);
	delete[] temp;
	auto end = CUR_TIME;
	LOG("Image Flip: (%d bit) %s (%dx%d) : %lf(s)\n",
		ch * 8,
		mode == FLIP::VERTICAL ?
		"Vertical" :
		mode == FLIP::HORIZONTAL ?
		"Horizontal" :
		mode == FLIP::BOTH ?
		"Both" : "Unknown",
		w, h,
		ELAPSED_TIME(begin, end));
	return bOK;
}

bool ImgControl::Crop(unsigned char *src, unsigned char *dst, int w, int h, int ch, int x, int y, int neww, int newh)
{
	if (!src || !dst) return false;
	if (x < 0 || y < 0 || x + neww > w || y + newh > h) return false;

	auto begin = CUR_TIME;
	bool bOK = true;
	int nBytePerLine = ((neww * ch) + 3) & ~3;
	int nBytePerLine2 = ((w * ch) + 3) & ~3;
	int offsetX = x * ch; // fix

	memset(dst, 0, sizeof(unsigned char) * nBytePerLine * newh);
	int num_threads = 1;

#ifdef _OPENMP
	int yy;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(yy)
		for (yy = 0; yy < newh; yy++) {
#else
		for (int yy = 0; yy < newh; yy++) {
#endif
			int offset = yy * nBytePerLine;
			int offsetY = (y + yy) * nBytePerLine2;
			memcpy(&dst[offset], &src[offsetY + offsetX], sizeof(unsigned char) * ch * neww);
		}
#ifdef _OPENMP
	}
#endif	
	auto end = CUR_TIME;
	LOG("Image Crop (%d threads): (%d bit) (%dx%d) : %lf(s)\n",
		num_threads, ch * 8, neww, newh,
		ELAPSED_TIME(begin, end));
	return bOK;
}

bool ImgControl::GetSize(const char* path, unsigned int *size)
{
	auto begin = CUR_TIME;
	bool bOK = true;
	*size = 0;
	int num_threads = 1;
	FILE *fp;
	fopen_s(&fp, path, "rb");

	if (!fp) {
		bOK = false;
		goto RETURN;
	}
	fseek(fp, 0, SEEK_END);
	*size = ftell(fp);
	fclose(fp);

RETURN:
	auto end = CUR_TIME;
	LOG("Get Size (%d threads): (%d bytes)\n",
		num_threads, *size,
		ELAPSED_TIME(begin, end));

	return bOK;
}

char* ImgControl::GetExtension(const char* path)
{
	const char *ext = PathFindExtensionA(path) + 1;

	return (char *)ext;
}
