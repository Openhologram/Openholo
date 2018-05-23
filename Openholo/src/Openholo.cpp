#include "Openholo.h"

#include <windows.h>
#include <fileapi.h>

#include "sys.h"

Openholo::Openholo(void)
{
	initialize();
}

Openholo::~Openholo(void)
{
}

void Openholo::initialize(void)
{
	context_ = { 0 };
	p_hologram = nullptr;
}

int Openholo::save(const char * fname, uint8_t bitsperpixel, void* src, int pic_width, int pic_height)
{
	if (checkExtension(fname, ".ohf"))	// save as *.ohf
		return saveAsOhf(fname, bitsperpixel, src, pic_width, pic_height);
	else								// save as image file - (bmp)
	{
		if (checkExtension(fname, ".bmp"))		// when the extension is bmp
			return saveAsImg(fname, bitsperpixel, src, pic_width, pic_height);
		else									// when extension is not .ohf, .bmp - force bmp
		{
			char buf[256];
			memset(buf, 0x00, sizeof(char) * 256);
			sprintf(buf, "%s.bmp", fname);

			return saveAsImg(buf, bitsperpixel, src, pic_width, pic_height);
		}
	}
}

int Openholo::load(const char * fname, void* dst)
{
	if (p_hologram != nullptr)
	{
		free(p_hologram);
		p_hologram = nullptr;
	}

	// load as .ohf file
	if (checkExtension(fname, ".ohf"))
	{
		if (dst != nullptr)
			return loadAsOhf(fname, dst);
		else
			return loadAsOhf(fname, p_hologram);
	}
	// load image file
	else
	{
		if (checkExtension(fname, ".bmp"))
		{
			if (dst != nullptr)
				return loadAsImg(fname, dst);
			else
				return loadAsImg(fname, p_hologram);
		}
		else			// when extension is not .ohf, .bmp
		{
			// how to load another image file format?
		}
	}

	return 0;
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

int Openholo::saveAsOhf(const char * fname, uint8_t bitsperpixel, void* src, int pic_width, int pic_height)
{
	//OphContext tmp = getContext();

	//int _height = pic_width;
	//int _width = pic_height;

	//if (pic_width == 0)
	//	_width = tmp.pixel_number.v[0];
	//if ( pic_height == 0)
	//	_height = tmp.pixel_number.v[1];

	//int _pixelbytesize = _height * _width * bitsperpixel / 8;
	//int _filesize = _pixelbytesize + sizeof(ohfdata);

	//FILE *fp;
	//fopen_s(&fp, fname, "wb");
	//if (fp == nullptr) return -1;

	//ohfdata *ohf = (ohfdata*)calloc(1, sizeof(ohfdata));
	//memset(ohf, 0x00, sizeof(ohfdata));

	//ohf->file_header.signature[0] = 'O';
	//ohf->file_header.signature[1] = 'H';
	//ohf->file_header.signature[2] = 'F';

	//ohf->file_header.filesize = _filesize;
	//ohf->file_header.offsetbits = sizeof(ohfdata);

	//ohf->field_header.headersize = sizeof(ohffieldheader);

	//ohf->field_header.pixelnum_x = _width;
	//ohf->field_header.pixelnum_y = _height;

	//ohf->field_header.pixelpitch_x = tmp.pixel_pitch.v[0];
	//ohf->field_header.pixelpitch_y = tmp.pixel_pitch.v[1];

	//ohf->field_header.pitchunit[0] = 'm';
	//ohf->field_header.pitchunit[1] = 't';

	//ohf->field_header.colortype[0] = 'G';
	//ohf->field_header.colortype[1] = 'R';
	//ohf->field_header.colortype[2] = 'Y';

	//ohf->field_header.wavelength = tmp.lambda;
	//ohf->field_header.wavelengthunit[0] = 'm';
	//ohf->field_header.wavelengthunit[1] = 't';

	//ohf->field_header.complexfieldtype[0] = 'i';
	//ohf->field_header.complexfieldtype[1] = 'm';
	//ohf->field_header.complexfieldtype[2] = 'a';
	//ohf->field_header.complexfieldtype[3] = 'g';
	//ohf->field_header.complexfieldtype[4] = 'e';
	////ohf->field_header.complexfieldtype[5] = '\0';

	//ohf->field_header.fieldtype[0] = 'A';
	//ohf->field_header.fieldtype[1] = 'P';

	//ohf->field_header.imageformat[0] = 'b';
	//ohf->field_header.imageformat[1] = 'm';
	//ohf->field_header.imageformat[2] = 'p';
	////ohf->field_header.imageformat[3] = '\0';
	////ohf->field_header.imageformat[4] = '\0';
	////ohf->field_header.imageformat[5] = '\0';
	////ohf->field_header.imageformat[6] = '\0';
	////ohf->field_header.imageformat[7] = '\0';
	////ohf->field_header.imageformat[8] = '\0';
	////ohf->field_header.imageformat[9] = '\0';

	//fwrite(ohf, 1, sizeof(ohfdata), fp);

	//fwrite(src, 1, _pixelbytesize, fp);
	//fclose(fp);
	//free(ohf);

	return 1;
}

int Openholo::saveAsImg(const char * fname, uint8_t bitsperpixel, void* src, int pic_width, int pic_height)
{
	OphContext tmp = getContext();

	int _width = pic_width, _height = pic_height;

	if (pic_width == 0)
		_width = tmp.pixel_number[_X];

	if (pic_height == 0)
		_height = tmp.pixel_number[_Y];

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
	return 1;
}

int Openholo::loadAsOhf(const char * fname, void* dst)
{
	FILE *infile;
	fopen_s(&infile, fname, "rb");
	if (infile == nullptr) { printf("No such file"); return 0; }

	ohfheader hf;
	ohffieldheader fHeader;
	fread(&hf, sizeof(ohfheader), 1, infile);
	if (hf.signature[0] != 'O' || hf.signature[1] != 'H' || hf.signature[2] != 'F') { printf("Not OHF File"); return 0; }

	fread(&fHeader, sizeof(ohffieldheader), 1, infile);

	int n_x = context_.pixel_number.v[0];
	int n_y = context_.pixel_number.v[1];

	p_hologram = (oph::uchar*)calloc(1, sizeof(oph::uchar)*n_x*n_y);

	fread(dst, sizeof(oph::uchar*), fHeader.pixelnum_x * fHeader.pixelnum_y, infile);

	fclose(infile);

	return 0;
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
	free(p_hologram);
	p_hologram = nullptr;
}
