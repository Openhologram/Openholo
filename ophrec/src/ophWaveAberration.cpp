/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install, copy or use the software.
//
//
//                           License Agreement
//                For Open Source Digital Holographic Library
//
// Openholo library is free software;
// you can redistribute it and/or modify it under the terms of the BSD 2-Clause license.
//
// Copyright (C) 2017-2024, Korea Electronics Technology Institute. All rights reserved.
// E-mail : contact.openholo@gmail.com
// Web : http://www.openholo.org
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  1. Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holder or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// This software contains opensource software released under GNU Generic Public License,
// NVDIA Software License Agreement, or CUDA supplement to Software License Agreement.
// Check whether software you use contains licensed software.
//
//M*/

#include "ophWaveAberration.h"



inline double ophWaveAberration::factorial(double x)
{
	if (x == 0)
		return (1);
	else
		return (x == 1 ? x : x * factorial(x - 1));
}


ophWaveAberration::ophWaveAberration()
	: nOrder(0)
	, mFrequency(0)
	, complex_W(nullptr)
{
	
	cout << "ophWaveAberration Constructor" << endl;
	uint wavelength_num = 1;

	complex_H = new Complex<Real>*[wavelength_num];
	context_.wave_length = new Real[wavelength_num];
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
	for (size_t i=0; i < x_max; i++)
	{
		delete[] A[i];
	}
	delete[] A;
		
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
	auto start_time = CUR_TIME;
	const oph::Complex<Real> j(0,1);

	double wave_lambda = context_.wave_length[0]; // wavelength
	int z_max = sizeof(zernikeCoefficent)/sizeof(zernikeCoefficent[0]);
	double *ZC;
	ZC = zernikeCoefficent;


	double n, m;
	double dxa = context_.pixel_pitch[_X];  // Sampling interval in x axis of exit pupil
	double dya = context_.pixel_pitch[_Y];  // Sampling interval in y axis of exit pupil
    unsigned int xr = context_.pixel_number[_X]; 
	unsigned int yr = context_.pixel_number[_Y]; // Resolution in x, y axis of exit pupil

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
	for (int i = 0; i < z_max; i++)
	{
		if (ZC[i] != 0)
		{
			n = ceil((-3 + sqrt(9 + 8 * i)) / 2); // order of the radial polynomial term
			m = 2 * i - n * (n + 2); // frequency of the sinusoidal component

			Temp_W = calculateZernikePolynomial(n, m, xn, yn, d);

			for(size_t ii = 0; ii < length_xn; ii++)
			{
				for (size_t jj = 0; jj < length_yn; jj++)
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
	//WD[x][y]

	for (int i = 0; i < (int)length_xn; i++)
	{
		delete [] W[i];
		delete [] Temp_W[i];
	}
	delete[] W;
	delete[] Temp_W; 

	for (int i = 0; i < (int)xr; i++)
	{
		delete[] WS[i];
	}
	delete[] WS;

	for (int i = 0; i < (int)length_xnn; i++)
	{
		delete[] WT[i];
	}
	delete[] WT;

	complex_W = WD;

	for (unsigned int x = 0; x < xr; x++) {
		for (unsigned int y = 0; y < yr; y++) {
			complex_H[0][x + y * xr] = complex_W[x][y];
		}
	}

//	return WD;

	auto end_time = CUR_TIME;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf sec\n", during_time);
}


void ophWaveAberration::Free2D(oph::Complex<Real> ** doublePtr)
{
	for (int i = 0; i < (int)context_.pixel_number[_X]; i++)
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


	if (!checkExtension(fname, ".xml"))
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
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryDoubleText(&context_.wave_length[0]))
		return false;


	xml_element = xml_node->FirstChildElement("PixelPitchHor");
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryDoubleText(&context_.pixel_pitch[_X]))
		return false;

	xml_element = xml_node->FirstChildElement("PixelPitchVer");
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryDoubleText(&context_.pixel_pitch[_Y]))
		return false;

	xml_element = xml_node->FirstChildElement("ResolutionHor");
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryIntText(&context_.pixel_number[_X]))
		return false;

	xml_element = xml_node->FirstChildElement("ResolutionVer");
	if (!xml_element || tinyxml2::XML_SUCCESS != xml_element->QueryIntText(&context_.pixel_number[_Y]))
		return false;

	xml_element = xml_node->FirstChildElement("ZernikeCoeff");
	xml_attribute = xml_element->FirstAttribute();

	for(int i=0; i< 45; i++)
	{
		if (!xml_attribute || tinyxml2::XML_SUCCESS != xml_attribute->QueryDoubleValue(&zernikeCoefficent[i]))
			return false;
		xml_attribute=xml_attribute->Next();
		
	}

	pixelPitchX = context_.pixel_pitch[_X];
	pixelPitchY = context_.pixel_pitch[_Y];

	resolutionX = context_.pixel_number[_X];
	resolutionY = context_.pixel_number[_Y];

	waveLength = *context_.wave_length;

	Openholo::setPixelPitchOHC(vec2(context_.pixel_pitch[_X], context_.pixel_pitch[_Y]));
	Openholo::setPixelNumberOHC(ivec2(context_.pixel_number[_X], context_.pixel_number[_Y]));
	Openholo::setWavelengthOHC(context_.wave_length[0], oph::LenUnit::m);
	
	cout << "Wavelength:             " << context_.wave_length[0] << endl;
	cout << "PixelPitch(Horizontal): " << context_.pixel_pitch[_X] << endl;
	cout << "PixelPitch(Vertical):   " << context_.pixel_pitch[_Y] << endl;
	cout << "Resolution(Horizontal): " << context_.pixel_number[_X] << endl;
	cout << "Resolution(Vertical):   " << context_.pixel_number[_Y] << endl;
	cout << "Zernike Coefficient:    " << endl;
	for(int i=0; i<45; i++)
	{ 
		if (i!=0 && (i+1)%5 == 0)
			cout << "z["<<i<<"]="<< zernikeCoefficent[i]<<endl;
		else
			cout << "z[" << i << "]=" << zernikeCoefficent[i] <<"	";
		zernikeCoefficent[i] = zernikeCoefficent[i] * context_.wave_length[0];
	}
	int xr = context_.pixel_number[_X];
	int yr = context_.pixel_number[_Y];

	complex_H[0] = new Complex<Real>[xr * yr];
	
	return true;

}

void ophWaveAberration::saveAberration(const char* fname)
{
	ofstream fout(fname, ios_base::out | ios_base::binary);
	fout.write((char *)complex_W, context_.pixel_number[_X] * context_.pixel_number[_Y] * sizeof(oph::Complex<Real>));
	fout.close();
}

void ophWaveAberration::readAberration(const char* fname)
{

	complex_W = new oph::Complex<Real>*[context_.pixel_number[_X]];
	for (int i = 0; i < (int)context_.pixel_number[_X]; i++)
	complex_W[i] = new oph::Complex<Real>[context_.pixel_number[_Y]];

	ifstream fin(fname, ios_base::in | ios_base::binary);
	fin.read((char *)complex_W, context_.pixel_number[_X]*context_.pixel_number[_Y]);
	fin.close();
}

bool ophWaveAberration::loadAsOhc(const char * fname)
{
	if (!Openholo::loadAsOhc(fname)) {
		LOG("Failed load file");
		return false;
	}

	pixelPitchX = context_.pixel_pitch[_X];
	pixelPitchY = context_.pixel_pitch[_Y];

	int xr = resolutionX = context_.pixel_number[_X];
	int yr = resolutionY = context_.pixel_number[_Y];

	waveLength = context_.wave_length[0];
	for (int x = 0; x < xr; x++) {
		for (int y = 0; y < yr; y++) {
			complex_W[x][y] = complex_H[0][x + y * xr];
		}
	}

	return true;
}