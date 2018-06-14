#include "Openholo.h"

#include <windows.h>
#include <fileapi.h>

#include "sys.h"

Openholo::Openholo(void)
	: Base()
{
}

Openholo::~Openholo(void)
{
}

int Openholo::checkExtension(const char * fname, const char * ext)
{	
	//return	1	: the extension of "fname" and "ext" is the same
	//			0	: the extension of "fname" and "ext" is not the same

	std::string filename(fname);
	size_t pos = filename.find(ext);
	if (pos == std::string::npos)
		//when there is no search string
		return 0;
	else
		return 1;
}

int Openholo::saveAsImg(const char * fname, uint8_t bitsperpixel, void* src, int pic_width, int pic_height)
{
	LOG("Saving...%s...", fname);
	auto start = _cur_time;

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
	pbitmap->bitmapinfoheader.planes = _planes;
	pbitmap->bitmapinfoheader.bitsperpixel = bitsperpixel;
	pbitmap->bitmapinfoheader.compression = _compression;
	pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter;
	pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter;
	pbitmap->bitmapinfoheader.numcolorspallette = 256;
	fwrite(pbitmap, 1, sizeof(bitmap), fp);

	fwrite(src, 1, _pixelbytesize, fp);
	fclose(fp);
	free(pbitmap);

	auto end = _cur_time;

	auto during = ((std::chrono::duration<real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);

	return 1;
}

int Openholo::loadAsImg(const char * fname, void* dst)
{
	FILE *infile;
	fopen_s(&infile, fname, "rb");
	if (infile == nullptr) { printf("No such file"); return 0; }

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { printf("Not BMP File");  return 0; }

	fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	fseek(infile, hf.fileoffset_to_pixelarray, SEEK_SET);

	oph::uchar* img_tmp;
	if (hInfo.imagesize == 0)
	{
		img_tmp = (oph::uchar*)malloc(sizeof(oph::uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
		fread(img_tmp, sizeof(oph::uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), infile);
	}
	else 
	{
		img_tmp = (oph::uchar*)malloc(hInfo.imagesize);
		fread(img_tmp, sizeof(oph::uchar), hInfo.imagesize, infile);
	}
	fclose(infile);

	// data upside down
	int bytesperpixel = hInfo.bitsperpixel / 8;
	int rowsz = bytesperpixel * hInfo.width;

	for (oph::uint k = 0; k < hInfo.height*rowsz; k++)
	{
		int r = k / rowsz;
		int c = k % rowsz;
		((oph::uchar*)dst)[(hInfo.height - r - 1)*rowsz + c] = img_tmp[r*rowsz + c];
	}

	free(img_tmp);

	return 1;
}

int Openholo::getImgSize(int & w, int & h, int & bytesperpixel, const char * file_name)
{
	char bmpFile[256];
	sprintf_s(bmpFile, "%s", file_name);
	FILE *infile;
	fopen_s(&infile, bmpFile, "rb");
	if (infile == NULL) { printf("No Image File"); return 0; }

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') return 0;
	fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	//if (hInfo.bitsperpixel != 8) { printf("Bad File Format!!"); return 0; }

	w = hInfo.width;
	h = hInfo.height;
	bytesperpixel = hInfo.bitsperpixel / 8;

	fclose(infile);

	return 1;
}

void Openholo::imgScaleBilnear(unsigned char * src, unsigned char * dst, int w, int h, int neww, int newh)
{
	for (int y = 0; y < newh; y++)
	{
		for (int x = 0; x < neww; x++)
		{
			float gx = (x / (float)neww) * (w - 1);
			float gy = (y / (float)newh) * (h - 1);

			int gxi = (int)gx;
			int gyi = (int)gy;

			uint32_t a00 = src[gxi + 0 + gyi * w];
			uint32_t a01 = src[gxi + 1 + gyi * w];
			uint32_t a10 = src[gxi + 0 + (gyi + 1)*w];
			uint32_t a11 = src[gxi + 1 + (gyi + 1)*w];

			float dx = gx - gxi;
			float dy = gy - gyi;

			dst[x + y * neww] = int(a00 * (1 - dx)*(1 - dy) + a01 * dx*(1 - dy) + a10 * (1 - dx)*dy + a11 * dx*dy);

		}
	}
}

void Openholo::convertToFormatGray8(unsigned char * src, unsigned char * dst, int w, int h, int bytesperpixel)
{
	int idx = 0;
	unsigned int r = 0, g = 0, b = 0;
	for (int i = 0; i < w*h*bytesperpixel; i++)
	{
		unsigned int r = src[i + 0];
		unsigned int g = src[i + 1];
		unsigned int b = src[i + 2];
		dst[idx++] = (r + g + b) / 3;
		i += bytesperpixel - 1;
	}
}

void Openholo::ophFree(void)
{
}