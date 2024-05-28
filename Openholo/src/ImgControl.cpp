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

/// <summary>
/// Get bitmap pixel size.
/// </summary>
/// <param name="width">: [in] bitmap width</param>
/// <param name="height">: [in] bitmap height</param>
/// <param name="channel">: [in] bitmap channel</param>
/// <returns>bitmap pixel buffer size</returns>
uint64_t ImgControl::GetPixelSize(const uint32_t width, const uint32_t height, const uint8_t channel)
{
	return (((static_cast<uint64_t>(width) * channel) + 3) & ~3) * static_cast<uint64_t>(height);
}

/// <summary>
/// Resize the bitmap.
/// </summary>
/// <param name="src">: [in] source pixel buffer</param>
/// <param name="dst">: [out] destination pixel buffer</param>
/// <param name="w">: [in] source bitmap width </param>
/// <param name="h">: [in] source bitmap height</param>
/// <param name="neww">: [in] destination bitmap width</param>
/// <param name="newh">: [in] destination bitmap height</param>
/// <param name="ch">: [in] bitmap channel</param>
/// <returns>success: true, fail: false</returns>
bool ImgControl::Resize(const uint8_t* src, uint8_t* dst, const uint32_t w, const uint32_t h, 
	const uint32_t neww, const uint32_t newh, const uint8_t ch)
{
	if (src == nullptr) return false;
	if (ch != 1 && ch != 3) return false;

	uint64_t newsize;
	uint32_t nBytePerLine = ((w * ch) + 3) & ~3; // src
	uint32_t nNewBytePerLine = ((neww * ch) + 3) & ~3; // dst

	if (dst == nullptr)
	{
		newsize = GetPixelSize(neww, newh, ch);
		dst = new uint8_t[newsize];
	}
	else
		newsize = nNewBytePerLine * newh;

	for (uint32_t y = 0; y < newh; y++)
	{
		uint32_t dst_offset_y = y * nNewBytePerLine;
		float gy = (y / (float)newh) * (h - 1);
		int gyi = (int)gy;
		float dy = gy - gyi;

		for (uint32_t x = 0; x < neww; x++)
		{
			float gx = (x / (float)neww) * (w - 1);
			int gxi = (int)gx;

			uint32_t src_offset_x = gxi * ch;
			uint32_t src_offset_below = src_offset_x + gyi * nBytePerLine;
			uint32_t src_offset_above = src_offset_x + (gyi + 1) * nBytePerLine;
			uint32_t dst_offset_x = x * ch;
			uint32_t dst_offset = dst_offset_y + dst_offset_x;
			float dx = gx - gxi;
			float w1 = (1 - dx) * (1 - dy);
			float w2 = dx * (1 - dy);
			float w3 = (1 - dx) * dy;
			float w4 = dx * dy;

			if ((dst_offset + (ch - 1)) < newsize)
			{
				if (ch == 3)
				{
					// blue
					dst[dst_offset + 0] = int(
						src[src_offset_below + 0] * w1 +
						src[src_offset_below + 3] * w2 +
						src[src_offset_above + 0] * w3 +
						src[src_offset_above + 3] * w4
						);
					// green
					dst[dst_offset + 1] = int(
						src[src_offset_below + 1] * w1 +
						src[src_offset_below + 4] * w2 +
						src[src_offset_above + 1] * w3 +
						src[src_offset_above + 4] * w4
						);
					// red
					dst[dst_offset + 2] = int(
						src[src_offset_below + 2] * w1 +
						src[src_offset_below + 5] * w2 +
						src[src_offset_above + 2] * w3 +
						src[src_offset_above + 5] * w4
						);
				}
				else
				{
					// grayscale
					dst[dst_offset] = int(
						src[src_offset_below + 0] * w1 +
						src[src_offset_below + 1] * w2 +
						src[src_offset_above + 0] * w3 +
						src[src_offset_above + 1] * w4
						);
				}
			}

		}
	}
	return true;
}

/// <summary>
/// Rotate the bitmap.
/// </summary>
/// <param name="rotate">: [in] rotate angle.</param>
/// <param name="src">: [in] source pixel buffer</param>
/// <param name="dst">: [out] destination pixel buffer</param>
/// <param name="w">: [in] source bitmap width</param>
/// <param name="h">: [in] source bitmap height</param>
/// <param name="neww">: [in] destination bitmap width</param>
/// <param name="newh">: [in] destination bitmap height</param>
/// <param name="ch">: [in] bitmap channel</param>
/// <returns>success: true, fail: false</returns>
bool ImgControl::Rotate(const double rotate, const uint8_t* src, uint8_t* dst, const uint32_t w, const uint32_t h, 
	const uint32_t neww, const uint32_t newh, const uint8_t ch)
{
	if (src == nullptr) return false;
	if (ch != 1 && ch != 3) return false;

	bool bChangeSize = false;
	if (neww != w || newh != h) {
		bChangeSize = true;
	}

	uint64_t nImgSize = bChangeSize ? GetPixelSize(neww, newh, ch) : GetPixelSize(w, h, ch);

	if (dst == nullptr)
	{
		dst = new uint8_t[nImgSize];
	}

	uint8_t* temp = new uint8_t[nImgSize]; //src

	// step. 1
	if (bChangeSize) {
		Resize(src, temp, w, h, neww, newh, ch);
	}
	else {
		memcpy(temp, src, nImgSize);
	}

	uint32_t nBytePerLine = ((neww * ch) + 3) & ~3;
	double radian = RADIAN(rotate);
	double cc = cos(radian);
	double ss = sin(-radian);
	double centerX = (double)neww / 2.0;
	double centerY = (double)newh / 2.0;

	for (uint32_t y = 0; y < newh; y++) {
		uint32_t dstY = y * nBytePerLine;

		for (uint32_t x = 0; x < neww; x++) {
			int origX = (int)(centerX + ((double)y - centerY) * ss + ((double)x - centerX) * cc);
			int origY = (int)(centerY + ((double)y - centerY) * cc - ((double)x - centerX) * ss);

			uint8_t pixels[4] = { 0, };
			if ((origY >= 0 && origY < (int)newh) && (origX >= 0 && origX < (int)neww)) {
				int offsetX = origX * ch;
				int offsetY = origY * nBytePerLine;

				memcpy(pixels, &temp[offsetY + offsetX], sizeof(uint8_t) * ch);
			}
			memcpy(&dst[dstY + (x * ch)], pixels, sizeof(uint8_t) * ch);
		}
	}
	delete[] temp;
	return true;
}


/// <summary>
/// Flip the bitmap.
/// </summary>
/// <param name="rotate">: [in] flip mode.</param>
/// <param name="src">: [in] source pixel buffer</param>
/// <param name="dst">: [out] destination pixel buffer</param>
/// <param name="w">: [in] source bitmap width</param>
/// <param name="h">: [in] source bitmap height</param>
/// <param name="ch">: [in] bitmap channel</param>
/// <returns>success: true, fail: false</returns>
bool ImgControl::Flip(FLIP mode, const uint8_t* src, uint8_t* dst, const uint32_t w, const uint32_t h, const uint8_t ch)
{
	if (src == nullptr) return false;
	if (ch != 1 && ch != 3) return false;

	bool bOK = true;


	if (dst == nullptr)
	{
		dst = new uint8_t[GetPixelSize(w, h, ch)];
	}

	const uint32_t nBytePerLine = ((w * ch) + 3) & ~3;

	if (mode == FLIP::VERTICAL) {
		for (uint32_t y = 0; y < h; y++) {
			uint32_t offset = y * nBytePerLine;
			uint32_t offset2 = (h - y - 1) * nBytePerLine;
			for (uint32_t x = 0; x < w; x++) {
				memcpy(&dst[offset + (x * ch)], &src[offset2 + (x * ch)], sizeof(uint8_t) * ch);
			}
		}
	}
	else if (mode == FLIP::HORIZONTAL) {
		for (uint32_t y = 0; y < h; y++) {
			uint32_t offset = y * nBytePerLine;
			for (uint32_t x = 0; x < w; x++) {
				memcpy(&dst[offset + (x * ch)], &src[offset + ((w * ch) - ((x + 1) * ch))], sizeof(uint8_t) * ch);
			}
		}
	}
	else if (mode == FLIP::BOTH) {
		for (uint32_t y = 0; y < h; y++) {
			uint32_t offset = y * nBytePerLine;
			uint32_t offset2 = (h - y - 1) * nBytePerLine;
			for (uint32_t x = 0; x < w; x++) {
				memcpy(&dst[offset + (x * ch)], &src[offset2 + ((w * ch) - ((x + 1) * ch))], sizeof(uint8_t) * ch);
			}
		}
	}
	else {
		// do nothing
		bOK = false;
	}

	return bOK;
}

/// <summary>
/// Crop the bitmap.
/// </summary>
/// <param name="src">: [in] source pixel buffer</param>
/// <param name="dst">: [out] destination pixel buffer</param>
/// <param name="w">: [in] source bitmap width</param>
/// <param name="h">: [in] source bitmap height</param>
/// <param name="ch">: [in] bitmap channel</param>
/// <param name="start_x">: [in] crop start offset x</param>
/// <param name="start_y">: [in] crop start offset y</param>
/// <param name="end_x">: [in] crop end offset x</param>
/// <param name="end_y">: [in] crop end offset y</param>
/// <returns>success: true, fail: false</returns>
bool ImgControl::Crop(const uint8_t* src, uint8_t* dst, const uint32_t w, const uint32_t h, const uint8_t ch, 
	const uint32_t start_x, const uint32_t start_y, const uint32_t end_x, const uint32_t end_y)
{
	if (src == nullptr) return false;
	if (start_x < 0 || start_y < 0 || end_x > w || end_y > h ||
		start_x >= end_x || start_y >= end_y) return false;

	uint32_t neww = end_x - start_x;
	uint32_t newh = end_y - start_y;

	if (dst == nullptr)
	{
		uint64_t nImgSize = GetPixelSize(neww, newh, ch);
		dst = new uint8_t[nImgSize];
		memset(dst, 0, nImgSize);
	}

	int nBytePerLineDst = ((neww * ch) + 3) & ~3;
	int nBytePerLineSrc = ((w * ch) + 3) & ~3;
	int offsetX = start_x * ch; // fix

	for (uint32_t i = 0; i < newh; i++) {
		uint32_t offset = i * nBytePerLineDst;
		uint32_t offsetY = (start_y + i) * nBytePerLineSrc;
		memcpy(&dst[offset], &src[offsetY + offsetX], sizeof(uint8_t) * ch * neww);
	}
	return true;
}

/// <summary>
/// Get file size.
/// </summary>
/// <param name="path">: [in] file path</param>
/// <param name="size">: [out] file size</param>
/// <returns>success: true, fail: false</returns>
bool ImgControl::GetSize(const char* path, uint32_t* size)
{
	bool bOK = true;
	*size = 0;
	FILE* fp = fopen(path, "rb");
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
