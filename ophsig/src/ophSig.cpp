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
	int src1_total = static_cast<int>(src1.size());
	int src2_total = static_cast<int>(src1.size());

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
			fft_in[src.size[_X] * i + j][_RE] = src(i, j).real();
			fft_in[src.size[_X] * i + j][_IM] = src(i, j).imag();
		}
	}

	fftw_plan plan = fftw_plan_dft_2d(src.size[_X], src.size[_Y], fft_in, fft_out, sign, flag);

	fftw_execute(plan);
	if (sign == OPH_FORWARD)
	{
		for (int i = 0; i < src.size[_X]; i++) {
			for (int j = 0; j < src.size[_Y]; j++) {
				dst(i, j)._Val[_RE] = fft_out[src.size[_X] * i + j][_RE];
				dst(i, j)._Val[_IM] = fft_out[src.size[_X] * i + j][_IM];
			}
		}
	}
	else if (sign == OPH_BACKWARD)
	{
		for (int i = 0; i < src.size[_X]; i++) {
			for (int j = 0; j < src.size[_Y]; j++) {
				dst(i, j)._Val[_RE] = fft_out[src.size[_X] * i + j][_RE] / (src.size[_X] * src.size[_Y]);
				dst(i, j)._Val[_IM] = fft_out[src.size[_X] * i + j][_IM] / (src.size[_X] * src.size[_Y]);

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
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			int ti = i - src.size[_X] / 2; if (ti < 0) ti += src.size[_X];
			int tj = j - src.size[_Y] / 2; if (tj < 0) tj += src.size[_Y];
			dst(tj, ti)._Val[_RE] = src(j, i).real();
			dst(tj, ti)._Val[_IM] = src(j, i).imag();
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
			while ((Xq[j]) >(X[i + 1])) i++;
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
	else if (realtype == "bmp")
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

			for (int row = 0; row < _cfgSig.rows; row++)
			{
				for (int col = 0; col < _cfgSig.cols; col++)
				{
					realMat[0](col, row) = realdata[_cfgSig.rows*row + col];
					imagMat[0](col, row) = imagdata[_cfgSig.rows*row + col];
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

				for (int row = 0; row < _cfgSig.rows; row++)
				{
					for (int col = 0; col < _cfgSig.cols; col++)
					{
						realMat[z](col, row) = realdata[row*_cfgSig.rows + col];
						imagMat[z](col, row) = imagdata[row*_cfgSig.rows + col];
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
		int _width = _cfgSig.rows, _height = _cfgSig.cols;

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
			LOG("file not found");
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

			// same to matlab save
			for (int i = _width - 1; i >= 0; i--)
			{
				for (int j = 0; j < _height; j++)
				{
					if (ComplexH[0].mat[_width - i - 1][j]._Val[_RE] <= 0)
					{
						realdata[i*_width + j] = 0;
					}
					else
					{
						realdata[i*_width + j] = (uchar)(ComplexH[0].mat[_width - i - 1][j]._Val[_RE] * 255);
					}

					if (ComplexH[0].mat[_width - i - 1][j]._Val[_IM] <= 0)
					{
						imagdata[i*_width + j] = 0;
					}
					else
					{
						imagdata[i*_width + j] = (uchar)(ComplexH[0].mat[_width - i - 1][j]._Val[_IM] * 255);
					}
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
			}
			for (int i = _cfgSig.rows - 1; i >= 0; i--)
			{
				for (int j = 0; j < _cfgSig.cols; j++)
				{
					for (int z = 0; z < 3; z++)
					{
						realdata[3 * j + 3 * i * _cfgSig.rows + z] = (uchar)((ComplexH[z].mat[_cfgSig.rows - i - 1][j]._Val[_RE] - minVal) / (maxVal - minVal) * 255);
						imagdata[3 * j + 3 * i * _cfgSig.rows + z] = (uchar)((ComplexH[z].mat[_cfgSig.rows - i - 1][j]._Val[_IM] - iminVal) / (imaxVal - iminVal) * 255);
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

			for (int row = 0; row < ComplexH[0].size[_X]; row++)
			{
				for (int col = 0; col < ComplexH[0].size[_Y]; col++)
				{
					realdata[_cfgSig.rows*row + col] = ComplexH[0].mat[col][row]._Val[_RE];
					imagdata[_cfgSig.rows*row + col] = ComplexH[0].mat[col][row]._Val[_IM];
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

				for (int row = 0; row < _cfgSig.rows; row++)
				{
					for (int col = 0; col < _cfgSig.cols; col++)
					{
						realdata[row * _cfgSig.rows + col] = ComplexH[z].mat[row][col]._Val[_RE];
						imagdata[row * _cfgSig.rows + col] = ComplexH[z].mat[row][col]._Val[_IM];
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
	
	return true;
}

bool ophSig::sigConvertHPO() {
	
	return true;
}

bool ophSig::sigConvertCAC(double red, double green, double blue) {
	
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
	
	return true;
}

matrix<Complex<Real>> ophSig::propagationHolo(matrix<Complex<Real>> complexH, float depth) {
	
	return complexH;
}

double ophSig::sigGetParamAT() {

	return 1;
}

double ophSig::sigGetParamSF(float zMax, float zMin, int sampN, float th) {
	
	return 1;
}

void ophSig::ophFree(void) {

}