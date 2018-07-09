#include "Openholo.h"

#include <windows.h>
#include <fileapi.h>

#include "sys.h"
#include "include.h"

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
	auto start = CUR_TIME;

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
	pbitmap->bitmapinfoheader.planes = OPH_PLANES;
	pbitmap->bitmapinfoheader.bitsperpixel = bitsperpixel;
	pbitmap->bitmapinfoheader.compression = OPH_COMPRESSION;
	pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	pbitmap->bitmapinfoheader.ypixelpermeter = Y_PIXEL_PER_METER;
	pbitmap->bitmapinfoheader.xpixelpermeter = X_PIXEL_PER_METER;
	pbitmap->bitmapinfoheader.numcolorspallette = 256;
	fwrite(pbitmap, 1, sizeof(bitmap), fp);

	fwrite(src, 1, _pixelbytesize, fp);
	fclose(fp);
	free(pbitmap);

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);

	return 1;
}

int Openholo::loadAsImg(const char * fname, void* dst)
{
	FILE *infile;
	fopen_s(&infile, fname, "rb");
	if (infile == nullptr) { LOG("No such file"); return 0; }

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File");  return 0; }

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
	if (infile == NULL) { LOG("No Image File"); return 0; }

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

void Openholo::fft1(int n, Complex<Real>* in, Complex<Real>* out, int sign, uint flag)
{
	fftw_complex *fft_in, *fft_out;
	fftw_plan plan;

	fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
	fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);

	for (int i = 0; i < n; i++) {
		fft_in[i][_RE] = in[i].real();
		fft_in[i][_IM] = in[i].imag();
	}

	plan = fftw_plan_dft_1d(n, fft_in, fft_out, sign, flag);

	fftw_execute(plan);

	for (int i = 0; i < n; i++) {
		out[i][_RE] = fft_out[i][_RE];
		out[i][_IM] = fft_out[i][_IM];
	}

	fftw_destroy_plan(plan);
	fftw_free(fft_in);
	fftw_free(fft_out);
}

void Openholo::fft2(oph::ivec2 n, Complex<Real>* in, Complex<Real>* out, int sign, uint flag)
{
	int pnx = n[_X], pny = n[_Y];

	fftw_complex *fft_in, *fft_out;
	fftw_plan plan;

	fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny);
	fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny);


	in = new Complex<Real>[pnx * pny];
	out = new Complex<Real>[pnx * pny];

	for (int j = 0; j < pny; j++)
		for (int i = 0; i < pnx; i++)
			in[pnx * j + i][_RE] = 255 * i / pnx;
		
	for (int i = 0; i < pnx * pny; i++) {
		fft_in[i][_RE] = in[i].real();
		fft_in[i][_IM] = in[i].imag();
	}

	plan = fftw_plan_dft_2d(pnx, pny, fft_in, fft_out, sign, flag);

	fftw_execute(plan);

	for (int i = 0; i < pnx * pny; i++) {
		out[i][_RE] = fft_out[i][_RE];
		out[i][_IM] = fft_out[i][_IM];
	}

	fftw_destroy_plan(plan);
	fftw_free(fft_in);
	fftw_free(fft_out);
}

void Openholo::fft3(oph::ivec3 n, Complex<Real>* in, Complex<Real>* out, int sign, uint flag)
{
	int pnx = n[_X], pny = n[_Y], pnz = n[_Z];

	fftw_complex *fft_in, *fft_out;
	fftw_plan plan;

	fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny * pnz);
	fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny * pnz);

	for (int i = 0; i < pnx * pny * pnz; i++) {
		fft_in[i][_RE] = in[i].real();
		fft_in[i][_IM] = in[i].imag();
	}

	plan = fftw_plan_dft_3d(pnx, pny, pnz, fft_in, fft_out, sign, flag);

	fftw_execute(plan);

	for (int i = 0; i < pnx * pny * pnz; i++) {
		out[i][_RE] = fft_out[i][_RE];
		out[i][_IM] = fft_out[i][_IM];
	}

	fftw_destroy_plan(plan);
	fftw_free(fft_in);
	fftw_free(fft_out);
}

void Openholo::fftwShift(Complex<Real>* src, Complex<Real>* dst, int nx, int ny, int type, bool bNormalized)
{
	Complex<Real>* tmp = (Complex<Real>*)malloc(sizeof(Complex<Real>)*nx*ny);
	memset(tmp, 0, sizeof(Complex<Real>)*nx*ny);
	fftShift(nx, ny, src, tmp);

	//fftw_plan fft_plan_fwd = nullptr, fft_plan_bwd = nullptr;
	fftw_plan fft_plan;

	fftw_complex *in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);
	fftw_complex *out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);

	for (int i = 0; i < nx*ny; i++)
	{
		in[i][_RE] = tmp[i][_RE];
		in[i][_IM] = tmp[i][_IM];
	}
	
	fft_plan = fftw_plan_dft_2d(nx, ny, in, out, type, OPH_ESTIMATE);
	fftw_execute(fft_plan);

	//if (type == OPH_FORWARD)
	//	fftw_execute_dft(fft_plan_fwd, in, out);
	//else
	//	fftw_execute_dft(fft_plan_bwd, in, out);

	int normalF = 1;
	if (bNormalized) normalF = nx * ny;
	memset(tmp, 0, sizeof(Complex<Real>)*nx*ny);

	for (int k = 0; k < nx*ny; k++) {
		tmp[k][_RE] = out[k][_RE] / normalF;
		tmp[k][_IM] = out[k][_IM] / normalF;
	}
	fftw_free(in);
	fftw_free(out);

	memset(dst, 0, sizeof(Complex<Real>)*nx*ny);
	fftShift(nx, ny, tmp, dst);

	free(tmp);
}

void Openholo::fftShift(int nx, int ny, Complex<Real>* input, Complex<Real>* output)
{
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			int ti = i - nx / 2; if (ti < 0) ti += nx;
			int tj = j - ny / 2; if (tj < 0) tj += ny;

			output[ti + tj * nx] = input[i + j * nx];
		}
	}
}

void Openholo::ophFree(void)
{
}