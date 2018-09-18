#include "ophSig.h"

ophSig::ophSig(void) {

}


template<typename T>
inline void ophSig::absMat(matrix<Complex<T>>& src, matrix<T>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst.mat[i][j] = sqrt(src.mat[i][j]._Val[_RE] * src.mat[i][j]._Val[_RE] + src.mat[i][j]._Val[_IM] * src.mat[i][j]._Val[_IM]);
		}
	}
}


template<typename T>
inline void ophSig::absMat(matrix<T>& src, matrix<T>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst.mat[i][j] = abs(src.mat[i][j]);
		}
	}
}

template<typename T>
inline void ophSig::angleMat(matrix<Complex<T>>& src, matrix<T>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			angle(src(i, j), dst(i, j));
		}
	}
}

template<typename T>
inline void ophSig::conjMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst(i, j) = src(i, j).conj();

		}
	}
}

template<typename T>
inline void ophSig::expMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst.mat[i][j]._Val[_RE] = exp(src.mat[i][j]._Val[_RE]) * cos(src.mat[i][j]._Val[_IM]);
			dst.mat[i][j]._Val[_IM] = exp(src.mat[i][j]._Val[_RE]) * sin(src.mat[i][j]._Val[_IM]);
		}
	}
}

template<typename T>
inline void ophSig::expMat(matrix<T>& src, matrix<T>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst.mat[i][j] = exp(src.mat[i][j]);
		}
	}
}

template<typename T>
inline Real ophSig::maxOfMat(matrix<T>& src) {
	Real max = MIN_DOUBLE;
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			if (src(i, j) > max) max = src(i, j);
		}
	}
	return max;
}

template<typename T>
void ophSig::meshgrid(vector<T>& src1, vector<T>& src2, matrix<T>& dst1, matrix<T>& dst2)
{
	/*int src1_total = src1.size();
	int src2_total = src2.size();*/
	int src1_total = static_cast<int>(src1.size());
	int src2_total = static_cast<int>(src2.size());

	dst1.resize(src2_total, src1_total);
	dst2.resize(src2_total, src1_total);
	for (int i = 0; i < src1_total; i++)
	{
		for (int j = 0; j < src2_total; j++)
		{
			dst1(j, i) = src1.at(i);
			dst2(j, i) = src2.at(j);
		}
	}
}

template<typename T>
inline Real ophSig::minOfMat(matrix<T>& src) {
	Real min = MAX_DOUBLE;
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			if (src(i, j) < min) min = src(i, j);
		}
	}
	return min;
}

template<typename T>
void ophSig::fft1(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign, uint flag)
{
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	fftw_complex *fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * src.size[_Y]);
	fftw_complex *fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * src.size[_Y]);

	for (int i = 0; i < src.size[_Y]; i++) {
		fft_in[i][_RE] = src(0, i).real();
		fft_in[i][_IM] = src(0, i).imag();
	}

	fftw_plan plan = fftw_plan_dft_1d(src.size[_Y], fft_in, fft_out, sign, flag);

	fftw_execute(plan);
	if (sign == OPH_FORWARD)
	{
		for (int i = 0; i < src.size[_Y]; i++) {
			dst(0, i)._Val[_RE] = fft_out[i][_RE];
			dst(0, i)._Val[_IM] = fft_out[i][_IM];
		}
	}
	else if (sign == OPH_BACKWARD)
	{
		for (int i = 0; i < src.size[_Y]; i++) {
			dst(0, i)._Val[_RE] = fft_out[i][_RE] / src.size[_Y];
			dst(0, i)._Val[_IM] = fft_out[i][_IM] / src.size[_Y];
		}
	}

	fftw_destroy_plan(plan);
	fftw_free(fft_in);
	fftw_free(fft_out);
}
template<typename T>
void ophSig::fft2(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign, uint flag)
{
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}

	fftw_complex *fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * src.size[_X] * src.size[_Y]);
	fftw_complex *fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * src.size[_X] * src.size[_Y]);

	for (int i = 0; i < src.size[_X]; i++) {
		for (int j = 0; j < src.size[_Y]; j++) {
			fft_in[src.size[_Y] * i + j][_RE] = src(i, j).real();
			fft_in[src.size[_Y] * i + j][_IM] = src(i, j).imag();
		}
	}

	fftw_plan plan = fftw_plan_dft_2d(src.size[_X], src.size[_Y], fft_in, fft_out, sign, flag);

	fftw_execute(plan);
	if (sign == OPH_FORWARD)
	{
		for (int i = 0; i < src.size[_X]; i++) {
			for (int j = 0; j < src.size[_Y]; j++) {
				dst(i, j)._Val[_RE] = fft_out[src.size[_Y] * i + j][_RE];
				dst(i, j)._Val[_IM] = fft_out[src.size[_Y] * i + j][_IM];
			}
		}
	}
	else if (sign == OPH_BACKWARD)
	{
		for (int i = 0; i < src.size[_X]; i++) {
			for (int j = 0; j < src.size[_Y]; j++) {
				dst(i, j)._Val[_RE] = fft_out[src.size[_Y] * i + j][_RE] / (src.size[_X] * src.size[_Y]);
				dst(i, j)._Val[_IM] = fft_out[src.size[_Y] * i + j][_IM] / (src.size[_X] * src.size[_Y]);

			}
		}
	}

	fftw_destroy_plan(plan);
	fftw_free(fft_in);
	fftw_free(fft_out);
}
template<typename T>
void ophSig::fftShift(matrix<Complex<T>> &src, matrix<Complex<T>> &dst)
{
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	int xshift = src.size[_X] / 2;
	int yshift = src.size[_Y] / 2;
	for (int i = 0; i < src.size[_X]; i++)
	{
		int ii = (i + xshift) % src.size[_X];
		for (int j = 0; j < src.size[_Y]; j++)
		{
			int jj = (j + yshift) % src.size[_Y];
			dst.mat[ii][jj]._Val[_RE] = src.mat[i][j].real();
			dst.mat[ii][jj]._Val[_IM] = src.mat[i][j].imag();

		}

	}
}

vector<Real> ophSig::linspace(double first, double last, int len) {
	vector<Real> result;
	for (int i = 0; i < len; i++)
	{
		result.push_back(0);
	}
	double step = (last - first) / (len - 1);
	for (int i = 0; i < len; i++) { result[i] = first + i*step; }
	return result;
}

template<typename T>
inline void ophSig::meanOfMat(matrix<T> &src, double &dst)
{
	dst = 0;
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst += src(i, j);
		}
	}
	dst = dst / (src.size[_X] * src.size[_Y]);
}

template<typename T>
void ophSig::linInterp(vector<T> &X, matrix<Complex<T>> &src, vector<T> &Xq, matrix<Complex<T>> &dst)
{
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	int size = src.size[_Y];

	for (int i = 0, j = 0; j < dst.size[_Y]; j++)
	{
		if ((Xq[j]) >= (X[size - 2]))
		{
			i = size - 2;
		}
		else
		{
			while ((Xq[j]) > (X[i + 1])) i++;
		}
		dst(0, j)._Val[_RE] = src(0, i).real() + (src(0, i + 1).real() - src(0, i).real()) / (X[i + 1] - X[i]) * (Xq[j] - X[i]);
		dst(0, j)._Val[_IM] = src(0, i).imag() + (src(0, i + 1).imag() - src(0, i).imag()) / (X[i + 1] - X[i]) * (Xq[j] - X[i]);
	}

}


bool ophSig::load(const char *real, const char *imag, uint8_t bitpixel)
{
	string realname = real;
	string imagname = imag;
	int checktype = static_cast<int>(realname.rfind("."));
	matrix<Real> realMat[3], imagMat[3];

	std::string realtype = realname.substr(checktype + 1, realname.size());
	std::string imgtype = imagname.substr(checktype + 1, realname.size());

	if (realtype != imgtype) {
		LOG("failed : The data type between real and imaginary is different!\n");
		return false;
	}
	if (realtype == "bmp")
	{
		FILE *freal, *fimag;
		fileheader hf;
		bitmapinfoheader hInfo;
		fopen_s(&freal, realname.c_str(), "rb"); fopen_s(&fimag, imagname.c_str(), "rb");
		if (!freal)
		{
			LOG("real bmp file open fail!\n");
			return false;
		}
		if (!fimag)
		{
			LOG("imaginary bmp file open fail!\n");
			return false;
		}
		fread(&hf, sizeof(fileheader), 1, freal);
		fread(&hInfo, sizeof(bitmapinfoheader), 1, freal);
		fread(&hf, sizeof(fileheader), 1, fimag);
		fread(&hInfo, sizeof(bitmapinfoheader), 1, fimag);

		if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File!\n"); }
		if ((hInfo.height == 0) || (hInfo.width == 0))
		{
			LOG("bmp header is empty!\n");
			hInfo.height = _cfgSig.rows;
			hInfo.width = _cfgSig.cols;
			if (_cfgSig.rows == 0 || _cfgSig.cols == 0)
			{
				LOG("check your parameter file!\n");
				return false;
			}
		}
		if ((_cfgSig.rows != hInfo.height) || (_cfgSig.cols != hInfo.width)) {
			LOG("image size is different!\n");
			_cfgSig.rows = hInfo.height;
			_cfgSig.cols = hInfo.width;
			LOG("changed parameter of size %d x %d\n", _cfgSig.cols, _cfgSig.rows);
		}
		hInfo.bitsperpixel = bitpixel;
		if (bitpixel == 8)
		{
			rgbquad palette[256];
			fread(palette, sizeof(rgbquad), 256, freal);
			fread(palette, sizeof(rgbquad), 256, fimag);

			realMat[0].resize(hInfo.height, hInfo.width);
			imagMat[0].resize(hInfo.height, hInfo.width);
			ComplexH[0].resize(hInfo.height, hInfo.width);
		}
		else
		{
			realMat[0].resize(hInfo.height, hInfo.width);
			imagMat[0].resize(hInfo.height, hInfo.width);
			ComplexH[0].resize(hInfo.height, hInfo.width);

			realMat[1].resize(hInfo.height, hInfo.width);
			imagMat[1].resize(hInfo.height, hInfo.width);
			ComplexH[1].resize(hInfo.height, hInfo.width);

			realMat[2].resize(hInfo.height, hInfo.width);
			imagMat[2].resize(hInfo.height, hInfo.width);
			ComplexH[2].resize(hInfo.height, hInfo.width);
		}

		uchar* realdata = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
		uchar* imagdata = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));

		fread(realdata, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), freal);
		fread(imagdata, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), fimag);

		fclose(freal);
		fclose(fimag);

		for (int i = hInfo.height - 1; i >= 0; i--)
		{
			for (int j = 0; j < static_cast<int>(hInfo.width); j++)
			{
				for (int z = 0; z < (hInfo.bitsperpixel / 8); z++)
				{
					realMat[z](hInfo.height - i - 1, j) = (double)realdata[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
					imagMat[z](hInfo.height - i - 1, j) = (double)imagdata[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
				}
			}
		}
		LOG("file load complete!\n");

		free(realdata);
		free(imagdata);
	}
	else if (realtype == "bin")
	{
		if (bitpixel == 8)
		{

			ifstream freal(realname, ifstream::binary);
			ifstream fimag(imagname, ifstream::binary);
			realMat[0].resize(_cfgSig.rows, _cfgSig.cols); imagMat[0].resize(_cfgSig.rows, _cfgSig.cols);
			ComplexH[0].resize(_cfgSig.rows, _cfgSig.cols);
			int total = _cfgSig.rows*_cfgSig.cols;
			double *realdata = new double[total];
			double *imagdata = new double[total];
			int i = 0;
			freal.read(reinterpret_cast<char*>(realdata), sizeof(double) * total);
			fimag.read(reinterpret_cast<char*>(imagdata), sizeof(double) * total);

			for (int col = 0; col < _cfgSig.cols; col++)
			{
				for (int row = 0; row < _cfgSig.rows; row++)
				{
					realMat[0](row, col) = realdata[_cfgSig.rows*col + row];
					imagMat[0](row, col) = imagdata[_cfgSig.rows*col + row];
				}
			}

			freal.close();
			fimag.close();
			delete[]realdata;
			delete[]imagdata;
		}
		else if (bitpixel == 24)
		{
			realMat[0].resize(_cfgSig.rows, _cfgSig.cols);
			imagMat[0].resize(_cfgSig.rows, _cfgSig.cols);
			ComplexH[0].resize(_cfgSig.rows, _cfgSig.cols);

			realMat[1].resize(_cfgSig.rows, _cfgSig.cols);
			imagMat[1].resize(_cfgSig.rows, _cfgSig.cols);
			ComplexH[1].resize(_cfgSig.rows, _cfgSig.cols);

			realMat[2].resize(_cfgSig.rows, _cfgSig.cols);
			imagMat[2].resize(_cfgSig.rows, _cfgSig.cols);
			ComplexH[2].resize(_cfgSig.rows, _cfgSig.cols);

			int total = _cfgSig.rows*_cfgSig.cols;


			string RGB_name[] = { "_B","_G","_R" };
			double *realdata = new  double[total];
			double *imagdata = new  double[total];

			for (int z = 0; z < (bitpixel / 8); z++)
			{
				ifstream freal(strtok((char*)realname.c_str(), ".") + RGB_name[z] + "bin", ifstream::binary);
				ifstream fimag(strtok((char*)imagname.c_str(), ".") + RGB_name[z] + "bin", ifstream::binary);

				freal.read(reinterpret_cast<char*>(realdata), sizeof(double) * total);
				fimag.read(reinterpret_cast<char*>(imagdata), sizeof(double) * total);

				for (int col = 0; col < _cfgSig.cols; col++)
				{
					for (int row = 0; row < _cfgSig.rows; row++)
					{
						realMat[z](row, col) = realdata[_cfgSig.rows*col + row];
						imagMat[z](row, col) = imagdata[_cfgSig.rows*col + row];
					}
				}
				freal.close();
				fimag.close();
			}
			delete[] realdata;
			delete[] imagdata;
		}
	}
	else
	{
		LOG("wrong type\n");
	}

	//nomalization
	double realout, imagout;
	for (int z = 0; z < (bitpixel) / 8; z++)
	{
		meanOfMat(realMat[z], realout); meanOfMat(imagMat[z], imagout);
		realMat[z] / realout; imagMat[z] / imagout;
		absMat(realMat[z], realMat[z]);
		absMat(imagMat[z], imagMat[z]);
		realout = maxOfMat(realMat[z]); imagout = maxOfMat(imagMat[z]);
		realMat[z] / realout; imagMat[z] / imagout;
		realout = minOfMat(realMat[z]); imagout = minOfMat(imagMat[z]);
		realMat[z] - realout; imagMat[z] - imagout;

		for (int i = 0; i < _cfgSig.rows; i++)
		{
			for (int j = 0; j < _cfgSig.cols; j++)
			{
				ComplexH[z](i, j)._Val[_RE] = realMat[z](i, j);
				ComplexH[z](i, j)._Val[_IM] = imagMat[z](i, j);

			}
		}
	}
	LOG("data nomalization complete\n");

	return true;
}
bool ophSig::save(const char *real, const char *imag, uint8_t bitpixel)
{
	string realname = real;
	string imagname = imag;

	int checktype = static_cast<int>(realname.rfind("."));

	if (realname.substr(checktype + 1, realname.size()) == "bmp")
	{
		oph::uchar* realdata;
		oph::uchar* imagdata;
		int _pixelbytesize = 0;
		int _width = _cfgSig.cols, _height = _cfgSig.rows;

		if (bitpixel == 8)
		{
			_pixelbytesize = _height * _width;
		}
		else
		{
			_pixelbytesize = _height * _width * 3;
		}
		int _filesize = 0;


		FILE *freal, *fimag;
		fopen_s(&freal, realname.c_str(), "wb");
		fopen_s(&fimag, imagname.c_str(), "wb");

		if ((freal == nullptr) || (fimag == nullptr))
		{
			LOG("file not found\n");
			return FALSE;
		}

		if (bitpixel == 8)
		{
			realdata = (oph::uchar*)malloc(sizeof(oph::uchar) * _cfgSig.rows * _cfgSig.cols);
			imagdata = (oph::uchar*)malloc(sizeof(oph::uchar) * _cfgSig.rows * _cfgSig.cols);
			_filesize = _pixelbytesize + sizeof(bitmap);

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


			//// denormalization
			for (int i = _height - 1; i >= 0; i--)
			{
				for (int j = 0; j < _width; j++)
				{
					if (ComplexH[0].mat[_height - i - 1][j]._Val[_RE] < 0)
					{
						ComplexH[0].mat[_height - i - 1][j]._Val[_RE] = 0;
					}
					
					if (ComplexH[0].mat[_height - i - 1][j]._Val[_IM] < 0)
					{
						ComplexH[0].mat[_height - i - 1][j]._Val[_IM] = 0;
					}
				}
			}

			double minVal, iminVal, maxVal, imaxVal;
			for (int j = 0; j < ComplexH[0].size[_Y]; j++) {
				for (int i = 0; i < ComplexH[0].size[_X]; i++) {
					if ((i == 0) && (j == 0))
					{
						minVal = ComplexH[0](i, j)._Val[_RE];
						maxVal = ComplexH[0](i, j)._Val[_RE];
					}
					else {
						if (ComplexH[0](i, j)._Val[_RE] < minVal)
						{
							minVal = ComplexH[0](i, j).real();
						}
						if (ComplexH[0](i, j)._Val[_RE] > maxVal)
						{
							maxVal = ComplexH[0](i, j).real();
						}
					}
					if ((i == 0) && (j == 0)) {
						iminVal = ComplexH[0](i, j)._Val[_IM];
						imaxVal = ComplexH[0](i, j)._Val[_IM];
					}
					else {
						if (ComplexH[0](i, j)._Val[_IM] < iminVal)
						{
							iminVal = ComplexH[0](i, j)._Val[_IM];
						}
						if (ComplexH[0](i, j)._Val[_IM] > imaxVal)
						{
							imaxVal = ComplexH[0](i, j)._Val[_IM];
						}
					}
				}
			}
			for (int i = _height - 1; i >= 0; i--)
			{
				for (int j = 0; j < _width; j++)
				{
					realdata[i*_width + j] = (uchar)((ComplexH[0](_height - i - 1, j)._Val[_RE] - minVal) / (maxVal - minVal) * 255 + 0.5);
					imagdata[i*_width + j] = (uchar)((ComplexH[0](_height - i - 1, j)._Val[_IM] - iminVal) / (imaxVal - iminVal) * 255 + 0.5);
				}
			}

			



			pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
			pbitmap->bitmapinfoheader.width = _width;
			pbitmap->bitmapinfoheader.height = _height;
			pbitmap->bitmapinfoheader.planes = OPH_PLANES;
			pbitmap->bitmapinfoheader.bitsperpixel = bitpixel;
			pbitmap->bitmapinfoheader.compression = OPH_COMPRESSION;
			pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
			pbitmap->bitmapinfoheader.ypixelpermeter = 0;
			pbitmap->bitmapinfoheader.xpixelpermeter = 0;
			pbitmap->bitmapinfoheader.numcolorspallette = 256;

			fwrite(pbitmap, 1, sizeof(bitmap), freal);
			fwrite(realdata, 1, _pixelbytesize, freal);

			fwrite(pbitmap, 1, sizeof(bitmap), fimag);
			fwrite(imagdata, 1, _pixelbytesize, fimag);

			fclose(freal);
			fclose(fimag);
			free(pbitmap);
		}
		else
		{
			realdata = (oph::uchar*)malloc(sizeof(oph::uchar) * _cfgSig.rows * _cfgSig.cols * bitpixel / 3);
			imagdata = (oph::uchar*)malloc(sizeof(oph::uchar) * _cfgSig.rows * _cfgSig.cols * bitpixel / 3);
			_filesize = _pixelbytesize + sizeof(fileheader) + sizeof(bitmapinfoheader);

			fileheader *hf = (fileheader*)calloc(1, sizeof(fileheader));
			bitmapinfoheader *hInfo = (bitmapinfoheader*)calloc(1, sizeof(bitmapinfoheader));

			hf->signature[0] = 'B';
			hf->signature[1] = 'M';
			hf->filesize = _filesize;
			hf->fileoffset_to_pixelarray = sizeof(fileheader) + sizeof(bitmapinfoheader);

			double minVal, iminVal, maxVal, imaxVal;
			for (int z = 0; z < 3; z++)
			{
				for (int j = 0; j < ComplexH[0].size[_Y]; j++) {
					for (int i = 0; i < ComplexH[0].size[_X]; i++) {
						if ((i == 0) && (j == 0))
						{
							minVal = ComplexH[z](i, j)._Val[_RE];
							maxVal = ComplexH[z](i, j)._Val[_RE];
						}
						else {
							if (ComplexH[z](i, j)._Val[_RE] < minVal)
							{
								minVal = ComplexH[z](i, j)._Val[_RE];
							}
							if (ComplexH[z](i, j)._Val[_RE] > maxVal)
							{
								maxVal = ComplexH[z](i, j)._Val[_RE];
							}
						}
						if ((i == 0) && (j == 0)) {
							iminVal = ComplexH[z](i, j)._Val[_IM];
							imaxVal = ComplexH[z](i, j)._Val[_IM];
						}
						else {
							if (ComplexH[z](i, j)._Val[_IM] < iminVal)
							{
								iminVal = ComplexH[z](i, j)._Val[_IM];
							}
							if (ComplexH[z](i, j)._Val[_IM] > imaxVal)
							{
								imaxVal = ComplexH[z](i, j)._Val[_IM];
							}
						}
					}
				}

				for (int i = _height - 1; i >= 0; i--)
				{
					for (int j = 0; j < _width; j++)
					{
						realdata[3 * j + 3 * i * _width + z] = (uchar)((ComplexH[z](_height - i - 1, j)._Val[_RE] - minVal) / (maxVal - minVal) * 255);
						imagdata[3 * j + 3 * i * _width + z] = (uchar)((ComplexH[z](_height - i - 1, j)._Val[_IM] - iminVal) / (imaxVal - iminVal) * 255);

					}
				}
			}
			hInfo->dibheadersize = sizeof(bitmapinfoheader);
			hInfo->width = _width;
			hInfo->height = _height;
			hInfo->planes = OPH_PLANES;
			hInfo->bitsperpixel = bitpixel;
			hInfo->compression = OPH_COMPRESSION;
			hInfo->imagesize = _pixelbytesize;
			hInfo->ypixelpermeter = 0;
			hInfo->xpixelpermeter = 0;

			fwrite(hf, 1, sizeof(fileheader), freal);
			fwrite(hInfo, 1, sizeof(bitmapinfoheader), freal);
			fwrite(realdata, 1, _pixelbytesize, freal);

			fwrite(hf, 1, sizeof(fileheader), fimag);
			fwrite(hInfo, 1, sizeof(bitmapinfoheader), fimag);
			fwrite(imagdata, 1, _pixelbytesize, fimag);

			fclose(freal);
			fclose(fimag);
			free(hf);
			free(hInfo);
		}

		free(realdata);
		free(imagdata);
		std::cout << "file save bmp complete" << endl;
		return TRUE;
	}
	else if (realname.substr(checktype + 1, realname.size()) == "bin")
	{

		if (bitpixel == 8)
		{
			std::ofstream cos(realname, std::ios::binary);
			std::ofstream sin(imagname, std::ios::binary);

			if (!cos.is_open()) {
				printf("real file not found.\n");
				cos.close();
				return FALSE;
			}
			if (!sin.is_open()) {
				printf("imag file not found.\n");
				sin.close();
				return FALSE;
			}

			double *realdata = new  double[ComplexH[0].size[_X] * ComplexH[0].size[_Y]];
			double *imagdata = new  double[ComplexH[0].size[_X] * ComplexH[0].size[_Y]];

			for (int col = 0; col < ComplexH[0].size[_Y]; col++)
			{
				for (int row = 0; row < ComplexH[0].size[_X]; row++)
				{
					realdata[_cfgSig.rows*col + row] = ComplexH[0].mat[row][col]._Val[_RE];
					imagdata[_cfgSig.rows*col + row] = ComplexH[0].mat[row][col]._Val[_IM];
				}
			}

			cos.write(reinterpret_cast<const char*>(realdata), sizeof(double) * ComplexH[0].size[_X] * ComplexH[0].size[_Y]);
			sin.write(reinterpret_cast<const char*>(imagdata), sizeof(double) * ComplexH[0].size[_X] * ComplexH[0].size[_Y]);

			cos.close();
			sin.close();
			delete[]realdata;
			delete[]imagdata;
		}
		else if (bitpixel == 24)
		{
			std::string RGB_name[] = { "_B.", "_G.", "_R." };

			double *realdata = new  double[_cfgSig.rows * _cfgSig.cols];
			double *imagdata = new  double[_cfgSig.rows * _cfgSig.cols];

			for (int z = 0; z < 3; z++)
			{
				std::ofstream cos(strtok((char*)realname.c_str(), ".") + RGB_name[z] + "bin", std::ios::binary);
				std::ofstream sin(strtok((char*)imagname.c_str(), ".") + RGB_name[z] + "bin", std::ios::binary);

				if (!cos.is_open()) {
					LOG("real file not found.\n");
					cos.close();
					return FALSE;
				}

				if (!sin.is_open()) {
					LOG("imag file not found.\n");
					sin.close();
					return FALSE;
				}

				for (int col = 0; col < ComplexH[0].size[_Y]; col++)
				{
					for (int row = 0; row < ComplexH[0].size[_X]; row++)
					{
						realdata[_cfgSig.rows*col + row] = ComplexH[z].mat[row][col]._Val[_RE];
						imagdata[_cfgSig.rows*col + row] = ComplexH[z].mat[row][col]._Val[_IM];
					}
				}
				cos.write(reinterpret_cast<const char*>(realdata), sizeof(double) * _cfgSig.rows * _cfgSig.cols);
				sin.write(reinterpret_cast<const char*>(imagdata), sizeof(double) * _cfgSig.rows * _cfgSig.cols);

				cos.close();
				sin.close();
			}
			delete[]realdata;
			delete[]imagdata;
		}
		std::cout << "file save binary complete" << endl;
	}
	return TRUE;
}


bool ophSig::sigConvertOffaxis() {
	matrix<Real> x, y, H1;
	vector<Real> X, Y;
	matrix<Complex<Real>> expSource, exp, offh;
	x.resize(_cfgSig.rows, _cfgSig.cols);
	y.resize(_cfgSig.rows, _cfgSig.cols);
	expSource.resize(_cfgSig.rows, _cfgSig.cols);
	exp.resize(_cfgSig.rows, _cfgSig.cols);
	offh.resize(_cfgSig.rows, _cfgSig.cols);
	H1.resize(_cfgSig.rows, _cfgSig.cols);
	vector<Real> r = linspace(1, _cfgSig.cols, _cfgSig.cols);
	vector<Real> c = linspace(1, _cfgSig.rows, _cfgSig.rows);

	for (int i = 0; i < _cfgSig.cols; i++)
	{
		X.push_back(_cfgSig.width / (_cfgSig.cols - 1)*(r[i] - 1) - _cfgSig.width / 2);
	}
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		Y.push_back(_cfgSig.height / (_cfgSig.rows - 1)*(c[i] - 1) - _cfgSig.height / 2);
	}
	meshgrid(X, Y, x, y);

	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			expSource(i, j)._Val[_RE] = 0;
			expSource(i, j)._Val[_IM] = ((2 * M_PI) / _cfgSig.lambda)*((x(i, j) *sin(_angleX)) + (y(i, j) *sin(_angleY)));
		}
	}
	expMat(expSource, exp);
	offh = ComplexH[0].mulElem(exp);
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			H1(i, j) = offh(i, j)._Val[_RE];
		}
	}

	double out = minOfMat(H1);

	H1 - out;
	out = maxOfMat(H1);
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			ComplexH[0](i, j)._Val[_RE] = H1(i, j) / out;
			ComplexH[0](i, j)._Val[_IM] = 0;
		}
	}

	return true;
}

bool ophSig::sigConvertHPO() {
	matrix<Real> x, y; matrix<Complex<Real>> expSource, exp, F1, G1, OUT_H, HPO;
	vector<Real> X, Y;
	x.resize(_cfgSig.rows, _cfgSig.cols);
	y.resize(_cfgSig.rows, _cfgSig.cols);
	expSource.resize(_cfgSig.rows, _cfgSig.cols);
	exp.resize(_cfgSig.rows, _cfgSig.cols);
	F1.resize(_cfgSig.rows, _cfgSig.cols);
	G1.resize(_cfgSig.rows, _cfgSig.cols);
	OUT_H.resize(_cfgSig.rows, _cfgSig.cols);
	HPO.resize(_cfgSig.rows, _cfgSig.cols);

	float NA = _cfgSig.width / (2 * _cfgSig.z);
	float NA_g = NA*_redRate;

	vector<Real> r = linspace(1, _cfgSig.cols, _cfgSig.cols);
	vector<Real> c = linspace(1, _cfgSig.rows, _cfgSig.rows);
	for (int i = 0; i < _cfgSig.cols; i++)
	{
		X.push_back(2 * M_PI*(r[i] - 1) / _cfgSig.width - M_PI*(_cfgSig.cols - 1) / _cfgSig.width);
	}
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		Y.push_back(2 * M_PI*(c[i] - 1) / _cfgSig.height - M_PI*(_cfgSig.rows - 1) / _cfgSig.height);
	}

	meshgrid(X, Y, x, y);
	double sigmaf = (_cfgSig.z*_cfgSig.lambda) / (4 * M_PI);
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			expSource(i, j)._Val[_RE] = 0;
			expSource(i, j)._Val[_IM] = sigmaf*(y(i, j)*y(i, j));
		}
	}
	expMat(expSource, exp);
	fftShift(exp, F1);
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			expSource(i, j)._Val[_RE] = ((-M_PI*((_cfgSig.lambda / (2 * M_PI*NA_g))*(_cfgSig.lambda / (2 * M_PI*NA_g))))*((y(i, j)*y(i, j))));
			expSource(i, j)._Val[_IM] = 0;
		}
	}
	expMat(expSource, exp);
	fftShift(exp, G1);
	fft2(ComplexH[0], OUT_H, OPH_FORWARD);
	HPO = G1.mulElem(F1.mulElem(OUT_H));
	fft2(HPO, ComplexH[0], OPH_BACKWARD);
	return true;
}

bool ophSig::sigConvertCAC(double red, double green, double blue) {
	double lambda[3];
	matrix<Real> x, y;
	matrix<Complex<Real>> FFZP, exp, FH, conj, FH_CAC;
	vector<Real> X, Y;
	lambda[0] = blue;
	lambda[1] = green;
	lambda[2] = red;
	x.resize(_cfgSig.rows, _cfgSig.cols);
	y.resize(_cfgSig.rows, _cfgSig.cols);
	FFZP.resize(_cfgSig.rows, _cfgSig.cols);
	exp.resize(_cfgSig.rows, _cfgSig.cols);
	FH.resize(_cfgSig.rows, _cfgSig.cols);
	conj.resize(_cfgSig.rows, _cfgSig.cols);
	FH_CAC.resize(_cfgSig.rows, _cfgSig.cols);
	for (int z = 0; z < 3; z++)
	{
		double sigmaf = ((_foc[2] - _foc[z])*lambda[z]) / (4 * M_PI);
		vector<Real> r = linspace(1, _cfgSig.cols, _cfgSig.cols);
		vector<Real> c = linspace(1, _cfgSig.rows, _cfgSig.rows);
		for (int i = 0; i < ophSig::_cfgSig.cols; i++)
		{
			X.push_back(2 * M_PI*(r[i] - 1) / _radius - M_PI*(ophSig::_cfgSig.cols - 1) / _radius);
		}
		for (int i = 0; i < ophSig::_cfgSig.rows; i++)
		{
			Y.push_back(2 * M_PI*(c[i] - 1) / _radius - M_PI*(ophSig::_cfgSig.rows - 1) / _radius);
		}
		meshgrid(X, Y, x, y);
		for (int i = 0; i < ophSig::_cfgSig.rows; i++)
		{
			for (int j = 0; j < ophSig::_cfgSig.cols; j++)
			{
				FFZP(i, j)._Val[_RE] = 0;
				FFZP(i, j)._Val[_IM] = sigmaf*((x(i, j)*x(i, j)) + (y(i, j)*y(i, j)));
			}
		}
		expMat(FFZP, exp);
		fftShift(exp, FFZP);
		fft2(ComplexH[z], FH, OPH_FORWARD);
		conjMat(FFZP, conj);
		FH_CAC = FH.mulElem(conj);
		fft2(FH_CAC, ComplexH[z], OPH_BACKWARD);
	}
	return true;
}

bool ophSig::readConfig(const char* fname)
{
	LOG("Reading....%s...\n", fname);

	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;

	if (checkExtension(fname, ".xml") == 0)
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fname);
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

	(xml_node->FirstChildElement("rows"))->QueryIntText(&_cfgSig.rows);
	(xml_node->FirstChildElement("cols"))->QueryIntText(&_cfgSig.cols);
	(xml_node->FirstChildElement("width"))->QueryFloatText(&_cfgSig.width);
	(xml_node->FirstChildElement("height"))->QueryFloatText(&_cfgSig.height);
	(xml_node->FirstChildElement("wavelength"))->QueryDoubleText(&_cfgSig.lambda);
	(xml_node->FirstChildElement("NA"))->QueryFloatText(&_cfgSig.NA);
	(xml_node->FirstChildElement("z"))->QueryFloatText(&_cfgSig.z);
	(xml_node->FirstChildElement("angle_X"))->QueryFloatText(&_angleX);
	(xml_node->FirstChildElement("angle_Y"))->QueryFloatText(&_angleY);
	(xml_node->FirstChildElement("reduction_rate"))->QueryFloatText(&_redRate);
	(xml_node->FirstChildElement("radius_of_lens"))->QueryFloatText(&_radius);
	(xml_node->FirstChildElement("focal_length_R"))->QueryFloatText(&_foc[2]);
	(xml_node->FirstChildElement("focal_length_G"))->QueryFloatText(&_foc[1]);
	(xml_node->FirstChildElement("focal_length_B"))->QueryFloatText(&_foc[0]);


	return true;
}


bool ophSig::propagationHolo(float depth) {
	int index = 0;
	int Z = 0;
	double sigma;
	double sigmaf;
	oph::matrix<Real> kx, ky;
	oph::matrix<oph::Complex<Real>> dst3;
	oph::matrix<oph::Complex<Real>>FH, FHI;

	oph::matrix<oph::Complex<Real>> FFZP(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	oph::matrix<oph::Complex<Real>> FFZP2(ComplexH[0].size[_X], ComplexH[0].size[_Y]);

	FFZP2 * 0;

	sigma = M_PI / (_cfgSig.lambda * depth);
	sigmaf = (depth * _cfgSig.lambda) / (4 * M_PI);

	int row, col;
	row = ComplexH[0].size[_X];
	col = ComplexH[0].size[_Y];

	int size1[] = { ComplexH[0].size[_X] };
	int size2[] = { ComplexH[0].size[_Y] };

	vector<Real> r(ComplexH[0].size[_X]);
	vector<Real> c(ComplexH[0].size[_Y]);

	r = this->linspace(1, row, row);
	c = this->linspace(1, col, col);

	for (int i = 0; i < r.size(); i++)
	{
		r.at(i) = (2 * M_PI * (r.at(i) - 1) / _cfgSig.height - M_PI*(row - 1) / _cfgSig.height);
	}

	for (int i = 0; i < c.size(); i++)
	{
		c.at(i) = ((2 * M_PI * (c.at(i) - 1)) / _cfgSig.width - M_PI*(col - 1) / (_cfgSig.width));
	}


	this->meshgrid(c, r, kx, ky);

	dst3.resize(kx.size[_X], kx.size[_Y]);

	for (int i = 0; i < dst3.size[_X]; i++)
	{
		for (int j = 0; j < dst3.size[_Y]; j++)
		{
			dst3(i, j)._Val[_RE] = 0;
			dst3(i, j)._Val[_IM] = sigmaf * ((kx(i, j) * kx(i, j) + ky(i, j) * ky(i, j)));
		}
	}

	expMat(dst3, FFZP);

	fftShift(FFZP, FFZP2);

	FH.resize(ComplexH[0].size[_X], ComplexH[0].size[_Y]);

	fft2(ComplexH[0], FH, OPH_FORWARD);

	FHI.resize(ComplexH[0].size[_X], ComplexH[0].size[_Y]);

	FHI = FH.mulElem(FFZP2);

	fft2(FHI, ComplexH[0], OPH_BACKWARD);

	return true;
}

matrix<Complex<Real>> ophSig::propagationHolo(matrix<Complex<Real>> complexH, float depth) {
	int index = 0;
	int Z = 0;
	double sigma;
	double sigmaf;
	oph::matrix<Real> kx, ky;
	oph::matrix<oph::Complex<Real>> dst3;
	oph::matrix<oph::Complex<Real>>FH, FHI;

	oph::matrix<oph::Complex<Real>> FFZP(complexH.size[_X], complexH.size[_Y]);
	oph::matrix<oph::Complex<Real>> FFZP2(complexH.size[_X], complexH.size[_Y]);

	FFZP2 * 0;

	sigma = M_PI / (_cfgSig.lambda * depth);
	sigmaf = (depth * _cfgSig.lambda) / (4 * M_PI);

	int row, col;
	row = complexH.size[_X];
	col = complexH.size[_Y];

	int size1[] = { complexH.size[_X] };
	int size2[] = { complexH.size[_Y] };

	vector<Real> r(complexH.size[_X]);
	vector<Real> c(complexH.size[_Y]);

	r = this->linspace(1, row, row);
	c = this->linspace(1, col, col);

	for (int i = 0; i < r.size(); i++)
	{
		r.at(i) = (2 * M_PI * (r.at(i) - 1) / _cfgSig.height - M_PI*(row - 1) / _cfgSig.height);
	}

	for (int i = 0; i < c.size(); i++)
	{
		c.at(i) = ((2 * M_PI * (c.at(i) - 1)) / _cfgSig.width - M_PI*(col - 1) / (_cfgSig.width));
	}


	this->meshgrid(c, r, kx, ky);

	dst3.resize(kx.size[_X], kx.size[_Y]);

	for (int i = 0; i < dst3.size[_X]; i++)
	{
		for (int j = 0; j < dst3.size[_Y]; j++)
		{
			dst3(i, j)._Val[_RE] = 0;
			dst3(i, j)._Val[_IM] = sigmaf * ((kx(i, j) * kx(i, j) + ky(i, j) * ky(i, j)));
		}
	}

	expMat(dst3, FFZP);

	fftShift(FFZP, FFZP2);

	FH.resize(complexH.size[_X], complexH.size[_Y]);

	fft2(complexH, FH);

	FHI.resize(complexH.size[_X], complexH.size[_Y]);

	FHI = FH.mulElem(FFZP2);

	this->fft2(FHI, complexH, OPH_BACKWARD);

	return complexH;
}

double ophSig::sigGetParamAT() {

	Real max = 0;	double index = 0;
	matrix<Complex<Real>> Flr(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	matrix<Complex<Real>> Fli(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	matrix<Complex<Real>> Hsyn(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	matrix<Complex<Real>> Hsyn_copy1(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	matrix<Complex<Real>> Hsyn_copy2(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	matrix<Real> Hsyn_copy3(ComplexH[0].size[_X], ComplexH[0].size[_Y]);

	matrix<Complex<Real>> Fo(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	matrix<Complex<Real>> Fo1(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	matrix<Complex<Real>> Fon, yn, Ab_yn;

	matrix<Real> Ab_yn_half, kx, ky, temp, G;
	vector<Real> r, c;
	vector<Real> t, tn;

	r = ophSig::linspace(1, _cfgSig.rows, _cfgSig.rows);
	c = ophSig::linspace(1, _cfgSig.cols, _cfgSig.cols);

	for (int i = 0; i < r.size(); i++)
	{
		r.at(i) = (2 * M_PI*(r.at(i) - 1) / _cfgSig.height - M_PI*(_cfgSig.rows - 1) / _cfgSig.height);
	}

	for (int i = 0; i < c.size(); i++)
	{
		c.at(i) = (2 * M_PI*(c.at(i) - 1) / _cfgSig.width - M_PI*(_cfgSig.cols - 1) / _cfgSig.width);
	}
	meshgrid(c, r, kx, ky);

	float NA_g = (float)0.025;

	temp.resize(kx.size[_X], kx.size[_Y]);
	G.resize(temp.size[_X], temp.size[_Y]);

	for (int i = 0; i < temp.size[_X]; i++)
	{
		for (int j = 0; j < temp.size[_Y]; j++)
		{
			temp(i, j) = -M_PI * (_cfgSig.lambda / (2 * M_PI * NA_g)) * (_cfgSig.lambda / (2 * M_PI * NA_g)) * (kx(i, j) * kx(i, j) + ky(i, j) * ky(i, j));
		}
	}

	expMat(temp, G);

	for (int i = 0; i < ComplexH[0].size[_X]; i++)
	{
		for (int j = 0; j < ComplexH[0].size[_Y]; j++)
		{
			Flr(i, j)._Val[0] = ComplexH[0](i, j)._Val[0];
			Fli(i, j)._Val[0] = ComplexH[0](i, j)._Val[1];
			Flr(i, j)._Val[1] = 0;
			Fli(i, j)._Val[1] = 0;
		}
	}

	fft2(Flr, Flr);
	fft2(Fli, Fli);

	for (int i = 0; i < Hsyn.size[_X]; i++)
	{
		for (int j = 0; j < Hsyn.size[_Y]; j++)
		{
			Hsyn(i, j)._Val[_RE] = Flr(i, j)._Val[_RE] * G(i, j);
			Hsyn(i, j)._Val[_IM] = Fli(i, j)._Val[_RE] * G(i, j);
		}
	}

	for (int i = 0; i < Hsyn.size[_X]; i++)
	{
		for (int j = 0; j < Hsyn.size[_Y]; j++)
		{
			Hsyn_copy1(i, j) = Hsyn(i, j);
			Hsyn_copy2(i, j) = Hsyn_copy1(i, j) * Hsyn(i, j);
		}
	}

	absMat(Hsyn, Hsyn_copy3);
	Hsyn_copy3 = Hsyn_copy3.mulElem(Hsyn_copy3) + pow(10, -300);




	for (int i = 0; i < Hsyn_copy2.size[_X]; i++)
	{
		for (int j = 0; j < Hsyn_copy2.size[_Y]; j++)
		{
			Fo(i, j)._Val[0] = Hsyn_copy2(i, j)._Val[0] / Hsyn_copy3(i, j);
			Fo(i, j)._Val[1] = Hsyn_copy2(i, j)._Val[1] / Hsyn_copy3(i, j);
		}
	}


	fftShift(Fo, Fo1);

	t = linspace(0, 1, _cfgSig.rows / 2 + 1);

	tn.resize(t.size());

	for (int i = 0; i < tn.size(); i++)
	{
		tn.at(i) = pow(t.at(i), 0.5);
	}

	Fon.resize(1, Fo.size[_X] / 2 + 1);

	for (int i = 0; i < Fo.size[_X] / 2 + 1; i++)
	{
		Fon(0, i)._Val[0] = Fo1(_cfgSig.rows / 2 - 1, _cfgSig.rows / 2 - 1 + i)._Val[0];
		Fon(0, i)._Val[1] = 0;
	}

	yn.resize(1, static_cast<int>(tn.size()));
	linInterp(t, Fon, tn, yn);
	fft1(yn, yn);
	Ab_yn.resize(yn.size[_X], yn.size[_Y]);
	absMat(yn, Ab_yn);
	Ab_yn_half.resize(1, _cfgSig.rows / 4 + 1);

	for (int i = 0; i < _cfgSig.rows / 4 + 1; i++)
	{
		Ab_yn_half(0, i) = Ab_yn(0, _cfgSig.rows / 4 + i - 1)._Val[_RE];
	}



	max = maxOfMat(Ab_yn_half);

	for (int i = 0; i < Ab_yn_half.size[1]; i++)
	{
		if (Ab_yn_half(0, i) == max)
		{
			index = i;
			break;
		}
	}

	index = -(((index + 1) - 120) / 10) / 140 + 0.1;

	return index;
}



double ophSig::sigGetParamSF(float zMax, float zMin, int sampN, float th) {

	matrix<Complex<Real>> I(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	vector<Real> F, z;
	F = linspace(1, sampN, sampN + 1);
	z = linspace(1, sampN, sampN + 1);
	float dz = (zMax - zMin) / sampN;
	Real max = MIN_DOUBLE;
	int index = 0;

	for (int n = 0; n < sampN + 1; n++)
	{
		matrix<Complex<Real>> F_I(ComplexH[0].size[_X], ComplexH[0].size[_Y]);

		for (int i = 0; i < F_I.size[_X]; i++)
		{
			for (int j = 0; j < F_I.size[_Y]; j++)
			{
				F_I(i, j) = 0;
			}
		}

		F.at(n) = 0;
		z.at(n) = -((n)* dz + zMin);

		I = propagationHolo(ComplexH[0], static_cast<float>(z.at(n)));

		for (int i = 0; i < I.size[_X] - 2; i++)
		{
			for (int j = 0; j < I.size[_Y] - 2; j++)
			{
				if (abs(I(i + 2, j)._Val[0] - I(i, j)._Val[0]) >= th)
				{
					F_I(i, j)._Val[0] = abs(I(i + 2, j)._Val[0] - I(i, j)._Val[0]) * abs(I(i + 2, j)._Val[0] - I(i, j)._Val[0]);
				}
				else if (abs(I(i, j + 2)._Val[0] - I(i, j)._Val[0]) >= th)
				{
					F_I(i, j)._Val[0] = abs(I(i, j + 2)._Val[0] - I(i, j)._Val[0]) * abs(I(i, j + 2)._Val[0] - I(i, j)._Val[0]);
				}
				F.at(n) += F_I(i, j)._Val[0];
			}
		}
		cout << (float)n / sampN * 100 << " %" << endl;
	}

	max = F.at(0);
	for (int i = 0; i < F.size(); i++) {
		if (F.at(i) > max) {
			max = F.at(i);
			index = i;
		}
	}

	return -z.at(index);
}

bool ophSig::getComplexHFromPSDH(const char * fname0, const char * fname90, const char * fname180, const char * fname270)
{
	string fname0str = fname0;
	string fname90str = fname90;
	string fname180str = fname180;
	string fname270str = fname270;
	int checktype = static_cast<int>(fname0str.rfind("."));
	matrix<Real> f0Mat[3], f90Mat[3], f180Mat[3], f270Mat[3];

	std::string f0type = fname0str.substr(checktype + 1, fname0str.size());

	uint8_t bitsperpixel;

	if (f0type == "bmp")
	{
		FILE *f0, *f90, *f180, *f270;
		fileheader hf;
		bitmapinfoheader hInfo;
		fopen_s(&f0, fname0str.c_str(), "rb"); fopen_s(&f90, fname90str.c_str(), "rb");
		fopen_s(&f180, fname180str.c_str(), "rb"); fopen_s(&f270, fname270str.c_str(), "rb");
		if (!f0)
		{
			LOG("bmp file open fail! (phase shift = 0)\n");
			return false;
		}
		if (!f90)
		{
			LOG("bmp file open fail! (phase shift = 90)\n");
			return false;
		}
		if (!f180)
		{
			LOG("bmp file open fail! (phase shift = 180)\n");
			return false;
		}
		if (!f270)
		{
			LOG("bmp file open fail! (phase shift = 270)\n");
			return false;
		}
		fread(&hf, sizeof(fileheader), 1, f0);
		fread(&hInfo, sizeof(bitmapinfoheader), 1, f0);

		if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File!\n"); }
		if ((hInfo.height == 0) || (hInfo.width == 0))
		{
			LOG("bmp header is empty!\n");
			hInfo.height = _cfgSig.rows;
			hInfo.width = _cfgSig.cols;
			if (_cfgSig.rows == 0 || _cfgSig.cols == 0)
			{
				LOG("check your parameter file!\n");
				return false;
			}
		}
		if ((_cfgSig.rows != hInfo.height) || (_cfgSig.cols != hInfo.width)) {
			LOG("image size is different!\n");
			_cfgSig.rows = hInfo.height;
			_cfgSig.cols = hInfo.width;
			LOG("changed parameter of size %d x %d\n", _cfgSig.cols, _cfgSig.rows);
		}
		bitsperpixel = hInfo.bitsperpixel;
		if (hInfo.bitsperpixel == 8)
		{
			f0Mat[0].resize(hInfo.height, hInfo.width);
			f90Mat[0].resize(hInfo.height, hInfo.width);
			f180Mat[0].resize(hInfo.height, hInfo.width);
			f270Mat[0].resize(hInfo.height, hInfo.width);
			ComplexH[0].resize(hInfo.height, hInfo.width);
		}
		else
		{
			f0Mat[0].resize(hInfo.height, hInfo.width);
			f90Mat[0].resize(hInfo.height, hInfo.width);
			f180Mat[0].resize(hInfo.height, hInfo.width);
			f270Mat[0].resize(hInfo.height, hInfo.width);
			ComplexH[0].resize(hInfo.height, hInfo.width);

			f0Mat[1].resize(hInfo.height, hInfo.width);
			f90Mat[1].resize(hInfo.height, hInfo.width);
			f180Mat[1].resize(hInfo.height, hInfo.width);
			f270Mat[1].resize(hInfo.height, hInfo.width);
			ComplexH[1].resize(hInfo.height, hInfo.width);

			f0Mat[2].resize(hInfo.height, hInfo.width);
			f90Mat[2].resize(hInfo.height, hInfo.width);
			f180Mat[2].resize(hInfo.height, hInfo.width);
			f270Mat[2].resize(hInfo.height, hInfo.width);
			ComplexH[2].resize(hInfo.height, hInfo.width);
		}

		uchar* f0data = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
		uchar* f90data = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
		uchar* f180data = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
		uchar* f270data = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));

		fread(f0data, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), f0);
		fread(f90data, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), f90);
		fread(f180data, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), f180);
		fread(f270data, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), f270);

		fclose(f0);
		fclose(f90);
		fclose(f180);
		fclose(f270);

		for (int i = hInfo.height - 1; i >= 0; i--)
		{
			for (int j = 0; j < static_cast<int>(hInfo.width); j++)
			{
				for (int z = 0; z < (hInfo.bitsperpixel / 8); z++)
				{
					f0Mat[z](hInfo.height - i - 1, j) = (double)f0data[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
					f90Mat[z](hInfo.height - i - 1, j) = (double)f90data[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
					f180Mat[z](hInfo.height - i - 1, j) = (double)f180data[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
					f270Mat[z](hInfo.height - i - 1, j) = (double)f270data[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
				}
			}
		}
		LOG("PSDH file load complete!\n");

		free(f0data);
		free(f90data);
		free(f180data);
		free(f270data);
		
	}
	else
	{
		LOG("wrong type (only BMP supported)\n");
	}

	// calculation complexH from 4 psdh and then normalize
	double normalizefactor = 1 / 256;
	for (int z = 0; z < (bitsperpixel / 8); z++)
	{
		for (int i = 0; i < _cfgSig.rows; i++)
		{
			for (int j = 0; j < _cfgSig.cols; j++)
			{
				ComplexH[z](i, j)._Val[_RE] = (f0Mat[z](i, j) - f180Mat[z](i,j))*normalizefactor;
				ComplexH[z](i, j)._Val[_IM] = (f90Mat[z](i, j) - f270Mat[z](i, j))*normalizefactor;

			}
		}
	}
	LOG("complex field obtained from 4 psdh\n");
	return true;
}

void ophSig::ophFree(void) {

}