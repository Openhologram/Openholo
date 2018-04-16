#ifndef __Bitmap_h
#define __Bitmap_h

#include "include.h"

//#define WIDTHBYTES(bits) (((bits)+31)/32*4)
//
////#define _bitsperpixel 8 //24 // 3바이트=24 
//#define _planes 1
//#define _compression 0
//#define _xpixelpermeter 0x130B //2835 , 72 DPI
//#define _ypixelpermeter 0x130B //2835 , 72 DPI
////#define pixel 0xFF
//
//#pragma pack(push,1)
//typedef struct{
//    uint8_t signature[2];
//    uint32_t filesize;
//    uint32_t reserved;
//    uint32_t fileoffset_to_pixelarray;
//} fileheader;
//typedef struct{
//    uint32_t dibheadersize;
//    uint32_t width;
//    uint32_t height;
//    uint16_t planes;
//    uint16_t bitsperpixel;
//    uint32_t compression;
//    uint32_t imagesize;
//    uint32_t ypixelpermeter;
//    uint32_t xpixelpermeter;
//    uint32_t numcolorspallette;
//    uint32_t mostimpcolor;
//} bitmapinfoheader;
//// 24비트 미만 칼라는 팔레트가 필요함
//typedef struct{
//    uint8_t rgbBlue;
//    uint8_t rgbGreen;
//    uint8_t rgbRed;
//    uint8_t rgbReserved;
//} rgbquad;
//typedef struct {
//    fileheader fileheader;
//    bitmapinfoheader bitmapinfoheader;
//	rgbquad rgbquad[256]; // 8비트 256칼라(흑백)
//} bitmap;
//#pragma pack(pop)

namespace oph
{
	//int createBitmapFile(unsigned char* pixelbuffer, int pic_width, int pic_height, uint16_t bitsperpixel, const char* file_name) {

	//	int _height = pic_height;
	//	int _width = pic_width;
	//	int _pixelbytesize = _height * _width*bitsperpixel / 8;
	//	int _filesize = _pixelbytesize + sizeof(bitmap);

	//	char bmpFile[256];
	//	std::string fname(file_name);
	//	int pos = fname.find(".bmp");
	//	if (pos == std::string::npos)
	//		sprintf_s(bmpFile, "%s.bmp", fname.c_str());
	//	else
	//	{
	//		if (strcmp(fname.substr(pos).c_str(), ".bmp") == 0)
	//			sprintf_s(bmpFile, "%s", fname.c_str());
	//		else
	//			sprintf_s(bmpFile, "%s.bmp", fname.c_str());
	//	}


	//	FILE *fp;
	//	fopen_s(&fp, bmpFile, "wb");
	//	bitmap *pbitmap = (bitmap*)calloc(1, sizeof(bitmap));
	//	memset(pbitmap, 0x00, sizeof(bitmap));

	//	// 파일헤더
	//	pbitmap->fileheader.signature[0] = 'B';
	//	pbitmap->fileheader.signature[1] = 'M';
	//	pbitmap->fileheader.filesize = _filesize;
	//	pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);

	//	// 팔레트 초기화: 흑백으로 만들어 줍니다.
	//	for (int i = 0; i < 256; i++) {
	//		pbitmap->rgbquad[i].rgbBlue = i;
	//		pbitmap->rgbquad[i].rgbGreen = i;
	//		pbitmap->rgbquad[i].rgbRed = i;
	//	}
	//	// 이미지 헤더
	//	pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	//	pbitmap->bitmapinfoheader.width = _width;
	//	pbitmap->bitmapinfoheader.height = _height;
	//	pbitmap->bitmapinfoheader.planes = _planes;
	//	pbitmap->bitmapinfoheader.bitsperpixel = bitsperpixel;
	//	pbitmap->bitmapinfoheader.compression = _compression;
	//	pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	//	pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter;
	//	pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter;
	//	pbitmap->bitmapinfoheader.numcolorspallette = 256;
	//	fwrite(pbitmap, 1, sizeof(bitmap), fp);

	//	// data upside down
	//	//unsigned char* img_tmp = (unsigned char*)malloc(sizeof(unsigned char)*_pixelbytesize);
	//	//int rowsz = _width * (bitsperpixel / 8);
	//	//for (int k = 0; k < _pixelbytesize; k++)
	//	//{
	//	//	int r = k / rowsz;
	//	//	int c = k % rowsz;
	//	//	img_tmp[(_height - r - 1)*rowsz + c] = pixelbuffer[r*rowsz + c];
	//	//}


	//	fwrite(pixelbuffer, 1, _pixelbytesize, fp);
	//	fclose(fp);
	//	free(pbitmap);
	//	return 1;
	//}

	//int getBitmapSize(int& w, int& h, int& bytesperpixel, const char* file_name)
	//{
	//	char bmpFile[256];
	//	sprintf_s(bmpFile, "%s", file_name);
	//	FILE *infile;
	//	fopen_s(&infile, bmpFile, "rb");
	//	if (infile == NULL) { printf("No Image File"); return 0; }

	//	// BMP Header Information
	//	fileheader hf;
	//	bitmapinfoheader hInfo;
	//	fread(&hf, sizeof(fileheader), 1, infile);
	//	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') return 0;
	//	fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	//	//if (hInfo.bitsperpixel != 8) { printf("Bad File Format!!"); return 0; }

	//	w = hInfo.width;
	//	h = hInfo.height;
	//	bytesperpixel = hInfo.bitsperpixel / 8;

	//	fclose(infile);

	//	return 1;

	//}

	//int loadBitmapFile(unsigned char* pixelbuffer, const char* file_name)
	//{
	//	char bmpFile[256];

	//	std::string fname(file_name);
	//	int pos = fname.find(".bmp");
	//	if (pos == std::string::npos)
	//		sprintf_s(bmpFile, "%s.bmp", fname.c_str());
	//	else
	//	{
	//		if (strcmp(fname.substr(pos).c_str(), ".bmp") == 0)
	//			sprintf_s(bmpFile, "%s", fname.c_str());
	//		else
	//			sprintf_s(bmpFile, "%s.bmp", fname.c_str());
	//	}

	//	FILE *infile;
	//	fopen_s(&infile, bmpFile, "rb");
	//	if (infile == NULL) { printf("No Image File"); return 0; }

	//	// BMP Header Information
	//	fileheader hf;
	//	bitmapinfoheader hInfo;
	//	fread(&hf, sizeof(fileheader), 1, infile);
	//	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') return 0;
	//	fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	//	//if (hInfo.bitsperpixel != 8) { printf("Bad File Format!!"); return 0; }

	//	// BMP Pallete
	//	//rgbquad hRGB[256];
	//	//fread(hRGB, sizeof(rgbquad), 256, infile);

	//	// Memory
	//	fseek(infile, hf.fileoffset_to_pixelarray, SEEK_SET);

	//	unsigned char* img_tmp;
	//	if (hInfo.imagesize == 0)
	//	{
	//		img_tmp = (unsigned char*)malloc(sizeof(unsigned char)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
	//		fread(img_tmp, sizeof(char), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), infile);
	//	}
	//	else {
	//		img_tmp = (unsigned char*)malloc(hInfo.imagesize);
	//		fread(img_tmp, sizeof(char), hInfo.imagesize, infile);
	//	}
	//	fclose(infile);

	//	// data upside down
	//	int bytesperpixel = hInfo.bitsperpixel / 8;
	//	int rowsz = bytesperpixel * hInfo.width;

	//	for (int k = 0; k < hInfo.height*rowsz; k++)
	//	{
	//		int r = k / rowsz;
	//		int c = k % rowsz;
	//		pixelbuffer[(hInfo.height - r - 1)*rowsz + c] = img_tmp[r*rowsz + c];
	//	}

	//	free(img_tmp);
	//	return 1;

	//}

	//void convertToFormatGray8(unsigned char* img, unsigned char* imgload, int w, int h, int bytesperpixel)
	//{
	//	int idx = 0;
	//	unsigned int r, g, b;
	//	for (int i = 0; i < w*h*bytesperpixel; i++)
	//	{
	//		unsigned int r = imgload[i + 0];
	//		unsigned int g = imgload[i + 1];
	//		unsigned int b = imgload[i + 2];
	//		img[idx++] = (r + g + b) / 3;
	//		i += bytesperpixel - 1;
	//	}
	//}

	//void imgScaleBilnear(unsigned char* src, unsigned char* dst, int w, int h, int neww, int newh)
	//{
	//	for (int y = 0; y < newh; y++)
	//	{
	//		for (int x = 0; x < neww; x++)
	//		{
	//			float gx = (x / (float)neww) * (w - 1);
	//			float gy = (y / (float)newh) * (h - 1);

	//			int gxi = (int)gx;
	//			int gyi = (int)gy;

	//			uint32_t a00 = src[gxi + 0 + gyi * w];
	//			uint32_t a01 = src[gxi + 1 + gyi * w];
	//			uint32_t a10 = src[gxi + 0 + (gyi + 1)*w];
	//			uint32_t a11 = src[gxi + 1 + (gyi + 1)*w];

	//			float dx = gx - gxi;
	//			float dy = gy - gyi;

	//			dst[x + y * neww] = a00 * (1 - dx)*(1 - dy) + a01 * dx*(1 - dy) + a10 * (1 - dx)*dy + a11 * dx*dy;

	//		}
	//	}
	//}
}
#endif // !__Bitmap_h