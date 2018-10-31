#include "ophSigCH.h"

ophSigCH::ophSigCH(void) 
	: Nz(0)
	, MaxIter(0)
	, Tau(0)
	, TolA(0)
	, TvIter(0)
{
}

bool ophSigCH::saveNumRec(const char *fname) {
	double gamma = 0.5;

	oph::uchar* intensityData;
	intensityData = (oph::uchar*)malloc(sizeof(oph::uchar) * _cfgSig.rows * _cfgSig.cols);
	double maxIntensity = 0.0;
	double realVal = 0.0;
	double imagVal = 0.0;
	double intensityVal = 0.0;
	for (int k = 0; k < Nz; k++)
	{
		for (int i = 0; i < _cfgSig.rows; i++)
		{
			for (int j = 0; j < _cfgSig.cols; j++)
			{
				realVal = NumRecRealImag(i, j + k*_cfgSig.cols);
				imagVal = NumRecRealImag(i + _cfgSig.rows, j + k*_cfgSig.cols);
				intensityVal = realVal*realVal + imagVal*imagVal;
				if (intensityVal > maxIntensity)
				{
					maxIntensity = intensityVal;
				}
			}
		}
	}

	string fnamestr = fname;
	int checktype = static_cast<int>(fnamestr.rfind("."));
	char str[10];

	for (int k = 0; k < Nz; k++)
	{
		fnamestr = fname;
		for (int i = _cfgSig.rows - 1; i >= 0; i--)
		{
			for (int j = 0; j < _cfgSig.cols; j++)
			{
				realVal = NumRecRealImag(i, j + k*_cfgSig.cols);
				imagVal = NumRecRealImag(i + _cfgSig.rows, j + k*_cfgSig.cols);
				intensityVal = realVal*realVal + imagVal*imagVal;
				intensityData[i*_cfgSig.cols + j] = (uchar)(pow(intensityVal / maxIntensity,gamma)*255.0);
			}
		}
		sprintf(str, "_%.2u", k);
		fnamestr.insert(checktype, str);
		saveAsImg(fnamestr.c_str(), 8, intensityData, _cfgSig.cols, _cfgSig.rows);
	}

	return TRUE;
}

bool ophSigCH::setCHparam(vector<Real> &z, int maxIter, double tau, double tolA, int tvIter) {
	MaxIter = maxIter;
	Tau = tau;
	TolA = tolA;
	TvIter = tvIter;
	Z.resize(z.size());
	Z = z;
	Nz = Z.size();
	return TRUE;
}

bool ophSigCH::runCH(int complexHidx) 
{
	matrix<Real> realimagplaneinput(_cfgSig.rows*2, _cfgSig.cols);
	c2ri(ComplexH[complexHidx], realimagplaneinput);

	NumRecRealImag.resize(_cfgSig.rows * 2, _cfgSig.cols*Nz);	
	twist(realimagplaneinput, NumRecRealImag);
	return TRUE;
}

void ophSigCH::tvdenoise(matrix<Real>& input, double lam, int iters, matrix<Real>& output)
{
	double dt = 0.25;
	int nr = input.size(_X);
	int nc = input.size(_Y);
	matrix<Real> divp(nr, nc);
	divp.zeros();
		
	matrix<Real> z(nr, nc);
	matrix<Real> z1, z2;
	z1.resize(nr, nc);
	z2.resize(nr, nc);

	matrix<Real> p1, p2;
	p1.resize(nr, nc);
	p2.resize(nr, nc);

	matrix<Real> denom(nr, nc);

	for (int iter = 0; iter < iters; iter++)
	{
		for (int i = 0; i < nr; i++)
		{
			for (int j = 0; j < nc; j++)
			{
				z(i,j) = divp(i, j) - input(i, j)*lam;
			}
		}

		/////////////////////////
		for (int i = 0; i < nr - 1; i++)
		{
			for (int j = 0; j < nc - 1; j++)
			{
				z1(i, j) = z(i, j + 1) - z(i, j);
				z2(i, j) = z(i + 1, j) - z(i, j);
				denom(i, j) = 1 + dt*sqrt(z1(i, j)*z1(i, j) + z2(i, j)*z2(i, j));
			}
		}
		for (int i = 0; i < nr-1; i++)
		{
			z1(i, nc-1) = 0.0;
			z2(i, nc-1) = z(i + 1, nc-1) - z(i, nc-1);
			denom(i, nc-1) = 1 + dt*sqrt(z1(i, nc-1)*z1(i, nc-1) + z2(i, nc-1)*z2(i, nc-1));
		}
		for (int j = 0; j < nc - 1; j++)
		{
			z1(nr-1, j) = z(nr-1, j+1) - z(nr-1, j);
			z2(nr-1, j) = 0.0;
			denom(nr-1, j) = 1 + dt*sqrt(z1(nr-1, j)*z1(nr-1, j) + z2(nr-1, j)*z2(nr-1, j));
		}
		denom(nr-1, nc-1) = 1.0;
		z1(nr - 1, nc - 1) = 0.0;
		z2(nr - 1, nc - 1) = 0.0;
		//////////////////////////


		for (int i = 0; i < nr ; i++)
		{
			for (int j = 0; j < nc; j++)
			{
				p1(i, j) = (p1(i, j) + dt*z1(i, j)) / denom(i, j);
				p2(i, j) = (p2(i, j) + dt*z2(i, j)) / denom(i, j);
			}
		}

		////////////////////////////////
		for (int i = 1; i < nr; i++)
		{
			for (int j = 1; j < nc; j++)
			{
				divp(i, j) = p1(i, j) - p1(i, j - 1) + p2(i, j) - p2(i - 1, j);
			}
		}
		for (int i = 1; i < nr; i++)
		{
			divp(i, 0) = p2(i, 0) - p2(i-1, 0);
		}
		for (int j = 1; j < nc; j++)
		{
			divp(0, j) = p1(0, j) - p1(0, j - 1);
		}
		divp(0, 0) = 0.0;
		//////////////////////////////

	}

	for (int i = 0; i < input.size[_X]; i++)
	{
		for (int j = 0; j < input.size[_Y]; j++)
		{
			output(i, j) = input(i,j) - divp(i, j)/lam;
		}
	}

}

double ophSigCH::tvnorm(matrix<Real>& input)
{
	double sqrtsum = 0.0;
	int nr = input.size[_X];
	int nc = input.size[_Y];

	for (int i = 0; i < nr-1; i++)
	{
		for (int j = 0; j < nc-1; j++)
		{
			sqrtsum += sqrt((input(i,j)-input(i+1,j))*(input(i,j)-input(i+1,j)) + (input(i,j)-input(i,j+1))*(input(i,j)-input(i,j+1)));
		}
	}
	for (int i = 0; i < nr-1; i++)
	{
		sqrtsum += sqrt(pow(input(i, nc-1) - input(i + 1, nc-1), 2) + pow(input(i, nc-1) - input(i, 0), 2));
	}
	for (int j = 0; j < nc - 1; j++)
	{
		sqrtsum += sqrt(pow(input(nr-1, j) - input(0, j), 2) + pow(input(nr-1, j) - input(nr-1, j+1), 2));
	}
	sqrtsum += sqrt(pow(input(nr-1, nc-1) - input(0, nc-1), 2) + pow(input(nr-1, nc-1) - input(nr-1, 0), 2));

	return sqrtsum;
}

void ophSigCH::c2ri(matrix<Complex<Real>>& complexinput, matrix<Real>& realimagoutput)
{
	for (int i = 0; i < complexinput.size[_X]; i++)
	{
		for (int j = 0; j < complexinput.size[_Y]; j++)
		{
			realimagoutput(i, j) = complexinput(i, j)._Val[_RE];
			realimagoutput(i + complexinput.size[_X], j) = complexinput(i, j)._Val[_IM];
		}
	}
	
}

void ophSigCH::ri2c(matrix<Real>& realimaginput, matrix<Complex<Real>>& complexoutput)
{
	for (int i = 0; i < complexoutput.size[_X]; i++)
	{
		for (int j = 0; j < complexoutput.size[_Y]; j++)
		{
			complexoutput(i, j)._Val[_RE] = realimaginput(i, j);
			complexoutput(i, j)._Val[_IM] = realimaginput(i + complexoutput.size[_X], j);
		}
	}
}

void ophSigCH::volume2plane(matrix<Real>& realimagvolumeinput, vector<Real> z, matrix<Real>& realimagplaneoutput)
{
	int nz = z.size();
	int nr = realimagvolumeinput.size(_X) / 2;	// real imag
	int nc = realimagvolumeinput.size(_Y) / nz;

	matrix<Complex<Real>> complexTemp(nr,nc);
	matrix<Complex<Real>> complexAccum(nr, nc);
	complexAccum.zeros();

	for (int k = 0; k < nz; k++)
	{
		for (int i = 0; i < nr; i++)
		{
			for (int j = 0; j < nc; j++)
			{
				complexTemp(i, j)._Val[_RE] = realimagvolumeinput(i, j + k*nc);
				complexTemp(i, j)._Val[_IM] = realimagvolumeinput(i + nr, j + k*nc);
			}
		}
		complexAccum = complexAccum + propagationHoloAS(complexTemp, static_cast<float>(z.at(k)));
	}
	c2ri(complexAccum, realimagplaneoutput);
}


void ophSigCH::plane2volume(matrix<Real>& realimagplaneinput, vector<Real> z, matrix<Real>& realimagplaneoutput)
{
	int nr = realimagplaneinput.size(_X)/2;
	int nc = realimagplaneinput.size(_Y);
	int nz = z.size();

	matrix<Complex<Real>> complexplaneinput(nr, nc);
	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			complexplaneinput(i, j)._Val[_RE] = realimagplaneinput(i, j);
			complexplaneinput(i, j)._Val[_IM] = realimagplaneinput(i + nr, j);
		}
	}


	matrix<Complex<Real>> temp(nr, nc);
	for (int k = 0; k < nz; k++)
	{
		temp = propagationHoloAS(complexplaneinput, static_cast<float>(-z.at(k)));
		for (int i = 0; i < nr; i++)
		{
			for (int j = 0; j < nc; j++)
			{
				realimagplaneoutput(i, j + k*nc) = temp(i, j)._Val[_RE];
				realimagplaneoutput(i + nr, j + k*nc) = temp(i, j)._Val[_IM];
			}
		}
	}

}

void ophSigCH::convert3Dto2D(matrix<Complex<Real>>* complex3Dinput, int nz, matrix<Complex<Real>>& complex2Doutput)
{
	int nr = complex3Dinput[0].size(_X);
	int nc = complex3Dinput[0].size(_Y);

	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			for (int k = 0; k < nz; k++)
			{
				complex2Doutput(i, j + k*nc) = complex3Dinput[k](i, j);
			}
		}
	}
}

void ophSigCH::convert2Dto3D(matrix<Complex<Real>>& complex2Dinput, int nz, matrix<Complex<Real>>* complex3Doutput)
{
	int nr = complex2Dinput.size(_X);
	int nc = complex2Dinput.size(_Y)/nz;

	for (int k = 0; k < nz; k++)
	{
		for (int i = 0; i < nr; i++)
		{
			for (int j = 0; j < nc; j++)
			{
				complex3Doutput[k](i, j) = complex2Dinput(i, j + k*nc);
			}
		}
	}
}

void ophSigCH::twist(matrix<Real>& realimagplaneinput, matrix<Real>& realimagvolumeoutput)
{
	//
	int nrv = realimagvolumeoutput.size(_X);
	int ncv = realimagvolumeoutput.size(_Y);
	int nrp = realimagplaneinput.size(_X);
	int ncp = realimagplaneinput.size(_Y);


	// TWiST
	double stopCriterion = 1;
	double tolA = TolA;
	int maxiter = MaxIter;
	int miniter = MaxIter;
	int init = 2;
	bool enforceMonotone = 1;
	bool compute_mse = 0;
	bool plot_ISNR = 0;
	bool verbose = 1;
	double alpha = 0;
	double beta = 0;
	bool sparse = 1;
	double tolD = 0.001;
	double phi_l1 = 0;
	double psi_ok = 0;
	double lam1 = 1e-4;
	double lamN = 1;
	double tau = Tau;
	int tv_iters = TvIter;

	bool for_ever = 1;
	double max_svd = 1;

	// twist parameters
	double rho0 = (1. - lam1 / lamN) / (1. + lam1 / lamN);
	if (alpha == 0) {
		alpha = 2. / (1 + sqrt(1 - pow(rho0, 2)));
	}
	if (beta == 0) {
		beta = alpha*2. / (lam1 + lamN);
	}

	double prev_f = 0.0;
	double f = 0.0;

	double criterion = 0.0;

	// initialization
	plane2volume(realimagplaneinput, Z, realimagvolumeoutput);

	// compute and sotre initial value of the objective function
	matrix<Real> resid(nrp, ncp);
	volume2plane(realimagvolumeoutput, Z, resid);
	for (int i = 0; i < nrp; i++)
	{
		for (int j = 0; j < ncp; j++)
		{
			resid(i, j) = realimagplaneinput(i, j) - resid(i, j);
		}
	}
	prev_f = 0.5*matrixEleSquareSum(resid) + tau*tvnorm(realimagvolumeoutput);

	//
	int iter = 0;
	bool cont_outer = 1;

	if (verbose)
	{
		LOG("initial objective = %10.6e\n", prev_f);
	}

	int IST_iters = 0;
	int TwIST_iters = 0;

	// initialize
	matrix<Real> xm1, xm2;
	xm1.resize(nrv, ncv);
	xm2.resize(nrv, ncv);
	xm1 = realimagvolumeoutput;
	xm2 = realimagvolumeoutput;

	// TwIST iterations
	matrix<Real> grad, temp_volume;
	grad.resize(nrv, ncv);
	temp_volume.resize(nrv, ncv);
	while (cont_outer)
	{
		plane2volume(resid, Z, grad);
		while (for_ever)
		{
			for (int i = 0; i < nrv; i++)
			{
				for (int j = 0; j < ncv; j++)
				{
					temp_volume(i, j) = xm1(i, j) + grad(i, j) / max_svd;
				}
			}
			tvdenoise(temp_volume, 2.0 / (tau / max_svd), tv_iters, realimagvolumeoutput);
			
			if ((IST_iters >= 2) | (TwIST_iters != 0))
			{
				if (sparse)
				{
					for (int i = 0; i < nrv; i++)
					{
						for (int j = 0; j < ncv; j++)
						{
							if (realimagvolumeoutput(i, j) == 0)
							{
								xm1(i, j) = 0.0;
								xm2(i, j) = 0.0;
							}
						}
					}
						
				}
				// two step iteration
				for (int i = 0; i < nrv; i++)
				{
					for (int j = 0; j < ncv; j++)
					{
						xm2(i, j) = (alpha - beta)*xm1(i, j) + (1.0 - alpha)*xm2(i, j) + beta*realimagvolumeoutput(i, j);
					}
				}
				// compute residual
				volume2plane(xm2, Z, resid);
				
				for (int i = 0; i < nrp; i++)
				{
					for (int j = 0; j < ncp; j++)
					{
						resid(i, j) = realimagplaneinput(i, j) - resid(i, j);
					}
				}
				f = 0.5*matrixEleSquareSum(resid) + tau*tvnorm(xm2);
				if ((f > prev_f) & (enforceMonotone))
				{
					TwIST_iters = 0;
				}
				else
				{
					TwIST_iters += 1;
					IST_iters = 0;
					realimagvolumeoutput = xm2;
					if (TwIST_iters % 10000 == 0)
					{
						max_svd = 0.9*max_svd;
					}
					break;
				}
			}
			else
			{
				volume2plane(realimagvolumeoutput, Z, resid);
				for (int i = 0; i < nrp; i++)
				{
					for (int j = 0; j < ncp; j++)
					{
						resid(i, j) = realimagplaneinput(i, j) - resid(i, j);
					}
				}
				f = 0.5*matrixEleSquareSum(resid) + tau*tvnorm(realimagvolumeoutput);
				if (f > prev_f)
				{
					max_svd = 2 * max_svd;
					if (verbose)
					{
						LOG("Incrementing S = %2.2e\n", max_svd);
					}
					IST_iters = 0;
					TwIST_iters = 0;
				}
				else
				{
					TwIST_iters += 1;
					break;
				}
			}

		}
		xm2 = xm1;
		xm1 = realimagvolumeoutput;

		criterion = (abs(f - prev_f)) / prev_f;

		cont_outer = ((iter <= maxiter) & (criterion > tolA));
		if (iter <= miniter)
		{
			cont_outer = 1;
		}

		iter += 1;
		prev_f = f;

		if (verbose)
		{
			LOG("Iteration=%4d, objective=%9.5e, criterion=%7.3e\n", iter, f, criterion / tolA);
		}
	}

	if (verbose)
	{
		double sum_abs_x=0.0;
		for (int i = 0; i < nrv; i++)
		{
			for (int j = 0; j < ncv; j++)
			{
				sum_abs_x += abs(realimagvolumeoutput(i, j));
			}
		}
		LOG("\n Finishied the main algorithm!\n");
		LOG("||Ax-y||_2 = %10.3e\n", matrixEleSquareSum(resid));
		LOG("||x||_1 = %10.3e\n", sum_abs_x);
		LOG("Objective function = %10.3e\n", f);
	}
}

double ophSigCH::matrixEleSquareSum(matrix<Real>& input)
{
	double output = 0.0;
	for (int i = 0; i < input.size(_X); i++)
	{
		for (int j = 0; j < input.size(_Y); j++)
		{
			output += input(i, j)*input(i, j);
		}
	}
	return output;
}

bool ophSigCH::readConfig(const char* fname)
{
	LOG("CH Reading....%s...\n", fname);

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
	(xml_node->FirstChildElement("wavelength"))->QueryDoubleText(&_cfgSig.wavelength[0]);
	(xml_node->FirstChildElement("nz"))->QueryIntText(&Nz);
	(xml_node->FirstChildElement("maxiter"))->QueryIntText(&MaxIter);
	(xml_node->FirstChildElement("tau"))->QueryDoubleText(&Tau);
	(xml_node->FirstChildElement("tolA"))->QueryDoubleText(&TolA);
	(xml_node->FirstChildElement("tv_iters"))->QueryIntText(&TvIter);

	double zmin, dz;
	(xml_node->FirstChildElement("zmin"))->QueryDoubleText(&zmin);
	(xml_node->FirstChildElement("dz"))->QueryDoubleText(&dz);
	
	Z.resize(Nz);
	for (int i = 0; i < Nz; i++)
	{
		Z.at(i) = zmin + i*dz;
	}
	
	return true;
}

bool ophSigCH::loadCHtemp(const char * real, const char * imag, uint8_t bitpixel)
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

	//////////////////////////////////////////////////////
	//////// From here, modified by Jae-Hyeung Park from original load function in ophSig 
	//nomalization 
	double realout, imagout;
	for (int z = 0; z < (bitpixel) / 8; z++)
	{
		for (int i = 0; i < _cfgSig.rows; i++)
		{
			for (int j = 0; j < _cfgSig.cols; j++)
			{
				ComplexH[z](i, j)._Val[_RE] = realMat[z](i, j)/255.0*2.0-1.0;
				ComplexH[z](i, j)._Val[_IM] = imagMat[z](i, j)/255.0*2.0-1.0;

			}
		}
	}
	LOG("data nomalization complete\n");

	return true;
}

matrix<Complex<Real>> ophSigCH::propagationHoloAS(matrix<Complex<Real>> complexH, float depth) {
	int nr = _cfgSig.rows;
	int nc = _cfgSig.cols;
	
	double dr = _cfgSig.height / _cfgSig.rows;
	double dc = _cfgSig.width / _cfgSig.cols;

	int nr2 = 2 * nr;
	int nc2 = 2 * nc;

	oph::matrix<oph::Complex<Real>> src2(nr2,nc2);	// prepare complex matrix with 2x size (to prevent artifacts caused by circular convolution)
	src2 * 0;	// initialize to 0

	int iStart = nr / 2 - 1;
	int jStart = nc / 2 - 1;
	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			src2(i+iStart, j+jStart)._Val[_RE] = complexH(i, j)._Val[_RE];
			src2(i+iStart, j+jStart)._Val[_IM] = complexH(i, j)._Val[_IM];

		}
	}

	double dfr = 1.0 / (((double)nr2)*dr);	// spatial frequency step in src2
	double dfc = 1.0 / (((double)nc2)*dc);

	matrix<Complex<Real>> propKernel(nr2, nc2);
	double fz = 0;
	for (int i = 0; i < nr2; i++)
	{
		for (int j = 0; j < nc2; j++)
		{
			fz = sqrt(pow(1.0 / _cfgSig.wavelength[0], 2) - pow((i - nr2 / 2.0 + 1.0)*dfr, 2) - pow((j - nc2 / 2.0 + 1.0)*dfc, 2));
			propKernel(i, j)._Val[_RE] = cos(2 * M_PI*depth*fz);
			propKernel(i, j)._Val[_IM] = sin(2 * M_PI*depth*fz);
		}
	}

	matrix<Complex<Real>> src2temp(nr2, nc2);
	matrix<Complex<Real>> FFZP(nr2, nc2);
	matrix<Complex<Real>> FFZP2(nr2, nc2);
	fftShift(src2, src2temp);
	fft2(src2temp, FFZP, OPH_FORWARD);
	fftShift(FFZP, FFZP2);
	FFZP2.mulElem(propKernel);

	fftShift(FFZP2, FFZP);
	fft2(FFZP, src2temp, OPH_BACKWARD);
	fftShift(src2temp, src2);

	matrix<Complex<Real>> dst(nr, nc);
	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			dst(i,j)._Val[_RE] = src2(i + iStart , j + jStart)._Val[_RE];
			dst(i,j)._Val[_IM] = src2(i + iStart , j + jStart)._Val[_IM];
		}
	}
	return dst;
}