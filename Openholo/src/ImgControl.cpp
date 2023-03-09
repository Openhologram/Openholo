#include "ImgControl.h"
#include <string.h>
#include <omp.h>
#include "sys.h"
#include "function.h"

ImgControl* ImgControl::instance = nullptr;

ImgControl::ImgControl()
{
}


ImgControl::~ImgControl()
{
}


unsigned long long ImgControl::GetBitmapSize(int width, int height, int channel)
{
	return (((width * channel) + 3) & ~3) * height;
}

void ImgControl::Resize(unsigned char* src, unsigned char* dst, int w, int h, int neww, int newh, int ch)
{
	if (src == nullptr) return;

	if (dst == nullptr)
	{
		dst = new unsigned char[GetBitmapSize(w, h, ch)];
	}

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
}

bool ImgControl::Rotate(double rotate, unsigned char *src, unsigned char *dst, int w, int h, int neww, int newh, int ch)
{
	if (src == nullptr) return false;
	if (ch > 4) return false;

	bool bChangeSize = false;
	if (neww != w || newh != h) {
		bChangeSize = true;
	}

	unsigned long long nImgSize = bChangeSize ? GetBitmapSize(neww, newh, ch) : GetBitmapSize(w, h, ch);

	if (dst == nullptr)
	{
		dst = new unsigned char[nImgSize];
	}
	
	unsigned char *temp = new unsigned char[nImgSize]; //src
	//unsigned char *temp2 = new unsigned char[nImgSize]; // dst

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
			//memcpy(&temp2[dstY + (x * ch)], pixels, sizeof(unsigned char) * ch);
			memcpy(&dst[dstY + (x * ch)], pixels, sizeof(unsigned char) * ch);
		}
	}

	//memcpy(dst, temp2, nImgSize);
	delete[] temp;
	//delete[] temp2;
	return true;
}

bool ImgControl::Flip(FLIP mode, unsigned char *src, unsigned char *dst, int w, int h, int ch)
{
	if (src == nullptr) return false;

	bool bOK = true;

	if (dst == nullptr)
	{
		dst = new unsigned char[GetBitmapSize(w, h, ch)];
	}

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
				memcpy(&dst[offset + (x * ch)], &src[offset2 + (x * ch)], sizeof(unsigned char) * ch);
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
				memcpy(&dst[offset + (x * ch)], &src[offset + ((w * ch) - ((x + 1) * ch))], sizeof(unsigned char) * ch);
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
				memcpy(&dst[offset + (x * ch)], &src[offset2 + ((w * ch) - ((x + 1) * ch))], sizeof(unsigned char) * ch);
			}
		}
	}
	else {
		// do nothing
		bOK = false;
	}

	return bOK;
}

bool ImgControl::Crop(unsigned char *src, unsigned char *dst, int w, int h, int ch, int x, int y, int neww, int newh)
{
	if (src == nullptr) return false;
	if (x < 0 || y < 0 || x + neww > w || y + newh > h) return false;

	if (dst == nullptr)
	{
		unsigned long long nImgSize = GetBitmapSize(neww, newh, ch);
		dst = new unsigned char[nImgSize];
		memset(dst, 0, nImgSize);
	}

	bool bOK = true;
	int nBytePerLine = ((neww * ch) + 3) & ~3;
	int nBytePerLine2 = ((w * ch) + 3) & ~3;
	int offsetX = x * ch; // fix


	int yy;
#ifdef _OPENMP
#pragma omp parallel for private(yy) firstprivate(nBytePerLine, nBytePerLine2, y, ch, neww)
	for (yy = 0; yy < newh; yy++) {
#endif
		int offset = yy * nBytePerLine;
		int offsetY = (y + yy) * nBytePerLine2;
		memcpy(&dst[offset], &src[offsetY + offsetX], sizeof(unsigned char) * ch * neww);
	}
	return bOK;
}

bool ImgControl::GetSize(const char* path, unsigned int *size)
{
	bool bOK = true;
	*size = 0;
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		bOK = false;
	}
	else
	{
		fseek(fp, 0, SEEK_END);
		*size = ftell(fp);
		fclose(fp);
	}
	return bOK;
}
