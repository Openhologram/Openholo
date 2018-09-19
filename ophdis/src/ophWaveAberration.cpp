#include "ophWaveAberration.h"



inline double ophWaveAberration::factorial(double x)
{
	if (x == 0)
		return (1);
	else
		return (x == 1 ? x : x * factorial(x - 1));
}


ophWaveAberration::ophWaveAberration() : nOrder(0), mFrequency(0)
{
	
	cout << "ophWaveAberration Constructor" << endl;
}

ophWaveAberration::~ophWaveAberration()
{
	cout << "ophWaveAberration Destructor" << endl;
}


double** ophWaveAberration::calculateZernikePolynomial(double n, double m, vector<double> x, vector<double> y, double d)
{
	vector<double>::size_type x_max = x.size();
	vector<double>::size_type y_max = y.size();
	double radius = d / 2;
	double N;
	double r;
	double theta;
	double co ;
	double si ;

	double **Z = new double*[x_max];
	double **A = new double*[x_max];
	for(int i = 0; i < (int)x_max; i++)
	{
		A[i] = new double[y_max];
		Z[i] = new double[y_max];

	//	memset(A[i], 0, y_max*sizeof(double));
	}

	for(int ix = 0; ix < (int)x_max; ix++)
	{ 
		for(int iy = 0; iy < (int)y_max; iy++)
		{ 
			A[ix][iy] = (sqrt(pow(x[ix],2) + pow(y[iy],2)) <= radius);
		};
	}
		// Start : Calculate Zernike polynomial

	N = sqrt(2 * (n + 1) / (1 + (m == 0))); // Calculate Normalization term

	if (n == 0)
	{
		for(int i=0; i<(int)x_max; i++)
		memcpy(Z[i], A[i],y_max*sizeof(double));
	}
	else
	{
		for(int i = 0; i<(int)x_max; i++)
			memset(Z[i],0, y_max*sizeof(double));

		for(int ix = 0; ix < (int)x_max; ix++)
		{
			for(int iy = 0; iy < (int)y_max; iy++)
			{ 
				r = sqrt(pow(x[ix], 2) + pow(y[iy],2));

				if (((x[ix] >= 0) && (y[iy] >= 0)) || ((x[ix] >= 0) & (y[iy] < 0)))
					theta = atan(y[iy] / (x[ix] + 1e-30));
				else
					theta = M_PI + atan(y[iy] / (x[ix] + 1e-30));
				
				for(int s = 0; s <= (n - abs(m)) / 2; s++)
				{ 
						Z[ix][iy] = Z[ix][iy] + pow((-1),s)*factorial(n - s)*pow((r/radius),(n - 2 * s)) /
						(factorial(s)*factorial((n + abs(m))/2 - s)*factorial((n - abs(m)) / 2 - s));
				}
				co = cos(m*theta);
				si = sin(m*theta);
				Z[ix][iy] = A[ix][iy]*N*Z[ix][iy]*((m >= 0)*co - (m < 0)*si);
			}
		}
	}
	// End : Calculate Zernike polynomial
	for (int i=0; i < x_max; i++)
	{
		delete[] A[i];
	}
	delete A;
		
	return Z;
}


void ophWaveAberration::imresize(double **X, int Nx, int Ny, int nx, int ny, double **Y)
{
	int fx, fy;
	double x, y, tx, tx1, ty, ty1, scale_x, scale_y;

	scale_x = (double)Nx / (double)nx;
	scale_y = (double)Ny / (double)ny;

	for (int i = 0; i < nx; i++) 
	{
		x = (double)i * scale_x;
	
		fx = (int)floor(x);
		tx = x - fx;
		tx1 = double(1.0) - tx;
		for (int j = 0; j < ny; j++)  
		{
			y = (double)j * scale_y;
			fy = (int)floor(y);
			ty = y - fy;
			ty1 = double(1.0) - ty;

			Y[i][j] = X[fx][fy] * (tx1*ty1) + X[fx][fy + 1] * (tx1*ty) + X[fx + 1][fy] * (tx*ty1) + X[fx + 1][fy + 1] * (tx*ty);
		}
	}
}



void ophWaveAberration::accumulateZernikePolynomial()
{
	const oph::Complex<Real> j(0,1);

	double wave_lambda = waveLength; // wavelength
	int z_max = sizeof(zernikeCoefficent)/sizeof(zernikeCoefficent[0]);
	double *ZC;
	ZC = zernikeCoefficent;


	double n, m;
	double dxa = pixelPitchX;  // Sampling interval in x axis of exit pupil
	double dya = pixelPitchY;  // Sampling interval in y axis of exit pupil
    unsigned int xr = resolutionX; 
	unsigned int yr = resolutionY; // Resolution in x, y axis of exit pupil

	double DE = max(dxa*xr, dya*yr);    // Diameter of exit pupil
	double scale = 1.3;

	DE = DE * scale;

	vector<double> xn;
	vector<double> yn;

	double max_xn = floor(DE/dxa+1);
	double max_yn = floor(DE/dya+1);

	xn.reserve((int)max_xn);
	for (int i = 0; i < (int)max_xn; i++)
	{
		xn.push_back(-DE / 2 + dxa*i);
	} // x axis coordinate of exit pupil

	yn.reserve((int)max_yn);
	for (int i = 0; i < max_yn; i++)
	{
		yn.push_back(-DE / 2 + dya*i);
	}// y axis coordinate of exit pupil
	
	double d = DE;

	vector<double>::size_type length_xn = xn.size();
	vector<double>::size_type length_yn = yn.size();

	double **W = new double*[(int)length_xn];
	double **Temp_W = new double*[(int)length_xn];

	for (int i = 0; i < (int)length_xn; i++)
	{
		W[i] = new double[length_yn];
		Temp_W[i] = new double[length_yn];
	}

	for (int i = 0; i < (int)length_xn; i++)
	{ 
		memset(W[i], 0, length_yn*sizeof(double));
		memset(Temp_W[i], 0, length_yn * sizeof(double));
	}


	// Start : Wavefront Aberration Generation
	for (int i = 0; i <z_max; i++)
	{
		if (ZC[i] != 0)
		{
			n = ceil((-3 + sqrt(9 + 8 * i)) / 2); // order of the radial polynomial term
			m = 2 * i - n * (n + 2); // frequency of the sinusoidal component

			Temp_W = calculateZernikePolynomial(n, m, xn, yn, d);

			for(int ii = 0; ii < length_xn; ii++)
			{
				for (int jj = 0; jj < length_yn; jj++) 
				{
					W[ii][jj] = W[ii][jj] + ZC[i] * Temp_W[ii][jj];
				}
			}
		}
	}
	// End : Wavefront Aberration Generation

	
	for (int i = 0; i < (int)length_xn; i++)
	{
		memset(Temp_W[i], 0, length_yn * sizeof(double));
	}
	
	int min_xnn, max_xnn;
	int min_ynn, max_ynn;
	
	min_xnn = (int)round(length_xn / 2 - xr / 2);
	max_xnn = (int)round(length_xn / 2 + xr / 2 + 1);
	min_ynn = (int)round(length_yn / 2 - yr / 2);
	max_ynn = (int)round(length_yn / 2 + yr / 2 + 1);

	int length_xnn, length_ynn;
	length_xnn = max_xnn - min_xnn;
	length_ynn = max_ynn - min_ynn;

	double **WT = new double*[length_xnn];
	for (int i = 0; i < length_xnn; i++)
	{
		WT[i] = new double[length_ynn];
	    memset(WT[i], 0, length_ynn * sizeof(double));
	}

	for (int i = 0; i < length_xnn; i++)
	{
		for (int j = 0; j < length_ynn; j++)
		{
			WT[i][j] = W[min_xnn+i][min_ynn+j];
		}
	}

	double **WS = new double*[(int)xr];
	for (int i = 0; i < (int)xr; i++)
	{
		WS[i] = new double[yr];
	    memset(WS[i], 0, yr * sizeof(double));
	}

	imresize(WT, length_xnn, length_ynn, xr, yr, WS);


	oph::Complex<Real> **WD = new oph::Complex<Real>*[xr];

	for(int i = 0; i < (int)xr; i++) 
		WD[i] = new oph::Complex<Real>[yr];

	for(int ii = 0; ii < (int)xr; ii ++ )
	{
		for (int jj = 0; jj < (int)yr; jj++)
		{
			
			WD[ii][jj]= exp(-j * (oph::Complex<Real>)2 * M_PI*WS[ii][jj] / wave_lambda);   // Wave Aberration Complex Field
		}
	}
	

	for (int i = 0; i < (int)length_xn; i++)
	{
		delete [] W[i];
		delete [] Temp_W[i];
	}
	delete W;
	delete Temp_W; 

	for (int i = 0; i < (int)xr; i++)
	{
		delete[] WS[i];
	}
	delete WS;

	for (int i = 0; i < (int)length_xnn; i++)
	{
		delete[] WT[i];
	}
	delete WT;

	complex_W = WD;



//	return WD;
}


void ophWaveAberration::Free2D(oph::Complex<Real> ** doublePtr)
{
	for (int i = 0; i < (int)resolutionX; i++)
	{
		delete[] doublePtr[i];
	}
}

void ophWaveAberration::ophFree(void)
{
	this->Free2D(complex_W);
	std::cout << " ophFree" << std::endl;
}


bool ophWaveAberration::readConfig(const char* fname)
{
	LOG("Reading....%s...\n", fname);

	
	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;
	tinyxml2::XMLElement *xml_element;
	const tinyxml2::XMLAttribute *xml_attribute;


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
	xml_element = xml_node->FirstChildElement("Wavelength");
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryDoubleText(&waveLength))
		return false;

	setWavelength(waveLength, oph::LenUnit::m);

	xml_element = xml_node->FirstChildElement("PixelPitchHor");
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryDoubleText(&pixelPitchX))
		return false;

	xml_element = xml_node->FirstChildElement("PixelPitchVer");
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryDoubleText(&pixelPitchY))
		return false;

	setPixelPitch(vec2(pixelPitchX, pixelPitchY));

	xml_element = xml_node->FirstChildElement("ResolutionHor");
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryUnsignedText(&resolutionX))
		return false;

	xml_element = xml_node->FirstChildElement("ResolutionVer");
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryUnsignedText(&resolutionY))
		return false;

	setPixelNumber(ivec2(resolutionX, resolutionY));

	xml_element = xml_node->FirstChildElement("ZernikeCoeff");
	xml_attribute = xml_element->FirstAttribute();

	for(int i=0; i< 45; i++)
	{
		if (!xml_attribute || tinyxml2::XML_SUCCESS != xml_attribute->QueryDoubleValue(&zernikeCoefficent[i]))
			return false;
		xml_attribute=xml_attribute->Next();
		
	}
	
	cout << "Wavelength:             " << waveLength << endl;
	cout << "PixelPitch(Horizontal): " << pixelPitchX << endl;
	cout << "PixelPitch(Vertical):   " << pixelPitchY << endl;
	cout << "Resolution(Horizontal): " << resolutionX << endl;
	cout << "Resolution(Vertical):   " << resolutionY << endl;
	cout << "Zernike Coefficient:    " << endl;
	for(int i=0; i<45; i++)
	{ 
		if (i!=0 && (i+1)%5 == 0)
			cout << "z["<<i<<"]="<< zernikeCoefficent[i]<<endl;
		else
			cout << "z[" << i << "]=" << zernikeCoefficent[i] <<"	";
		zernikeCoefficent[i] = zernikeCoefficent[i] * waveLength;
	}
	
	return true;

}

void ophWaveAberration::saveAberration(const char* fname)
{
	ofstream fout(fname, ios_base::out | ios_base::binary);
	fout.write((char *)complex_W, resolutionX * resolutionY * sizeof(oph::Complex<Real>));
	fout.close();
}

void ophWaveAberration::readAberration(const char* fname)
{

	complex_W = new oph::Complex<Real>*[resolutionX];
	for (int i = 0; i < (int)resolutionX; i++)
	complex_W[i] = new oph::Complex<Real>[resolutionY];

	ifstream fin(fname, ios_base::in | ios_base::binary);
	fin.read((char *)complex_W, resolutionX*resolutionY);
	fin.close();
}


