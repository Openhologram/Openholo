#include "ophSigConvert.h"

ophSigConvert::ophSigConvert(void) {

}

bool ophSigConvert::loadParam(std::string cfg) {

	std::ifstream inFile(cfg, std::ios::in);
	if (!inFile.is_open()) {
		printf("file not found.\n");
		inFile.close();
		return false;
	}
	std::string Line;
	std::stringstream LineStream;

	while (std::getline(inFile, Line)) {
		std::string title;
		std::string value;
		std::string equal; // " = "
		LineStream << Line;
		LineStream >> title >> equal >> value;
		if (strcmp(title.c_str(), "##") == 0) {
			LineStream.str("");
			LineStream.clear();
		}
		else {
			if (strcmp(title.c_str(), "rows") == 0) {
				_cfgSig.rows = stoi(value);
			}
			else if (strcmp(title.c_str(), "cols") == 0) {
				_cfgSig.cols = stoi(value);
			}
			else if (strcmp(title.c_str(), "width") == 0) {
				_cfgSig.width = stod(value);
			}
			else if (strcmp(title.c_str(), "height") == 0) {
				_cfgSig.height = stod(value);
			}
			else if (strcmp(title.c_str(), "wavelength_B") == 0) {
				_cfgSig.lambda[0] = stod(value);
			}
			else if (strcmp(title.c_str(), "wavelength_G") == 0) {
				_cfgSig.lambda[1] = stod(value);
			}
			else if (strcmp(title.c_str(), "wavelength_R") == 0) {
				_cfgSig.lambda[2] = stod(value);
			}
			else if (strcmp(title.c_str(), "NA") == 0) {
				_cfgSig.NA = stod(value);
			}
			else if (strcmp(title.c_str(), "z") == 0) {
				_cfgSig.z = stod(value);
			}
			else if (strcmp(title.c_str(), "angle_X") == 0) {
				_angleX = stof(value);
			}
			else if (strcmp(title.c_str(), "angle_Y") == 0) {
				_angleY = stof(value);
			}
			else if (strcmp(title.c_str(), "reduction_rate") == 0) {
				_redRate = stof(value);
			}
			else if (strcmp(title.c_str(), "radius_of_lens") == 0) {
				_radius = stof(value);
			}
			else if (strcmp(title.c_str(), "focal_length_B") == 0) {
				_foc[0] = stof(value);
			}
			else if (strcmp(title.c_str(), "focal_length_G") == 0) {
				_foc[1] = stof(value);
			}
			else if (strcmp(title.c_str(), "focal_length_R") == 0) {
				_foc[2] = stof(value);
			}
			LineStream.clear();
		}
	}

	inFile.close();
	return true;
}

bool ophSigConvert::sigConvertOffaxis() {
	int size[] = { ophSig::_cfgSig.rows,ophSig::_cfgSig.cols };
	cv::Mat buffer(2, size, CV_32FC2, cv::Scalar(0));
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			buffer.at<std::complex<float>>(i, j)._Val[0] = ophSig::complexH.at<std::complex<float>>(i, j, 0).real();
			buffer.at<std::complex<float>>(i, j)._Val[1] = ophSig::complexH.at<std::complex<float>>(i, j, 0).imag();
		}
	}
	cv::Mat r(1, ophSig::_cfgSig.rows, CV_32FC1, cv::Scalar(0));
	cv::Mat c(1, ophSig::_cfgSig.cols, CV_32FC1, cv::Scalar(0));
	r = ophSig::linspace(1, ophSig::_cfgSig.rows, ophSig::_cfgSig.rows);
	c = ophSig::linspace(1, ophSig::_cfgSig.cols, ophSig::_cfgSig.cols);

	cv::Mat X(1, ophSig::_cfgSig.rows, CV_32FC1, cv::Scalar(0));
	cv::Mat Y(1, ophSig::_cfgSig.cols, CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		X.at<float>(i) = (ophSig::_cfgSig.width / (ophSig::_cfgSig.rows - 1)*(r.at<float>(i) - 1) - ophSig::_cfgSig.width / 2);
	}
	for (int i = 0; i < ophSig::_cfgSig.cols; i++)
	{
		Y.at<float>(i) = (ophSig::_cfgSig.height / (ophSig::_cfgSig.cols - 1)*(c.at<float>(i) - 1) - ophSig::_cfgSig.height / 2);
	}
	cv::Mat x(2, size, CV_32FC1, cv::Scalar(0));
	cv::Mat y(2, size, CV_32FC1, cv::Scalar(0));

	meshgrid(X, Y, x, y);
	cv::Mat expSource(2, size, CV_32FC2, cv::Scalar(0));
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			expSource.at<std::complex<float>>(i, j)._Val[1] = ((2 * CV_PI) / ophSig::_cfgSig.lambda[0])*((x.at<float>(i, j)*std::sin(ophSigConvert::_angleX)) + (y.at<float>(i, j)*std::sin(ophSigConvert::_angleY)));
		}
	}
	cv::Mat exp1(2, size, CV_32FC2, cv::Scalar(0));
	exp(expSource, exp1);
	cv::Mat offh(2, size, CV_32FC2, cv::Scalar(0));
	ophSig::mul(buffer, exp1, offh);
	cv::Mat H1(2, size, CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			H1.at<float>(i, j) = offh.at<std::complex<float>>(i, j)._Val[0];
		}
	}

	float out;
	ophSig::min(H1, out);
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			H1.at<float>(i, j) = H1.at<float>(i, j) - out;
		}
	}
	ophSig::max(H1, out);
	H1 = H1 / out;
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			ophSig::complexH.at<std::complex<float>>(i, j, 0)._Val[0] = H1.at<float>(i, j);
			ophSig::complexH.at<std::complex<float>>(i, j, 0)._Val[1] = 0;
		}
	}
	return true;
}

bool ophSigConvert::sigConvertHPO() {
	float NA = ophSig::_cfgSig.width / (2 * ophSig::_cfgSig.z);
	float NA_g = NA*_redRate;
	int size[] = { ophSig::_cfgSig.rows,ophSig::_cfgSig.cols };
	cv::Mat buffer(2, size, CV_32FC2, cv::Scalar(0));
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			buffer.at<std::complex<float>>(i, j)._Val[0] = complexH.at<std::complex<float>>(i, j, 0).real();
			buffer.at<std::complex<float>>(i, j)._Val[1] = complexH.at<std::complex<float>>(i, j, 0).imag();
		}
	}
	cv::Mat r(1, ophSig::_cfgSig.rows, CV_32FC1, cv::Scalar(0));
	cv::Mat c(1, ophSig::_cfgSig.cols, CV_32FC1, cv::Scalar(0));
	r = linspace(1, ophSig::_cfgSig.rows, ophSig::_cfgSig.rows);
	c = linspace(1, ophSig::_cfgSig.cols, ophSig::_cfgSig.cols);
	cv::Mat X(1, ophSig::_cfgSig.rows, CV_32FC1, cv::Scalar(0));
	cv::Mat Y(1, ophSig::_cfgSig.cols, CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		X.at<float>(i) = (2 * CV_PI*(r.at<float>(i) - 1) / ophSig::_cfgSig.width - CV_PI*(ophSig::_cfgSig.rows - 1) / ophSig::_cfgSig.width);
	}
	for (int i = 0; i < ophSig::_cfgSig.cols; i++)
	{
		Y.at<float>(i) = (2 * CV_PI*(c.at<float>(i) - 1) / ophSig::_cfgSig.height - CV_PI*(ophSig::_cfgSig.cols - 1) / ophSig::_cfgSig.height);
	}

	cv::Mat x(2, size, CV_32FC1, cv::Scalar(0));
	cv::Mat y(2, size, CV_32FC1, cv::Scalar(0));
	meshgrid(X, Y, x, y);
	float sigmaf = (ophSig::_cfgSig.z*ophSig::_cfgSig.lambda[0]) / (4 * CV_PI);
	cv::Mat expSource(2, size, CV_32FC2, cv::Scalar(0));
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			expSource.at<std::complex<float>>(i, j)._Val[1] = sigmaf*(y.at<float>(i, j)*y.at<float>(i, j));
		}
	}
	cv::Mat F(2, size, CV_32FC2, cv::Scalar(0));
	exp(expSource, F);
	cv::Mat F1(2, size, CV_32FC2, cv::Scalar(0));
	fftshift(F, F1);
	cv::Mat expSource2(2, size, CV_32FC2, cv::Scalar(0));
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			expSource2.at<std::complex<float>>(i, j)._Val[0] = ((-CV_PI*((ophSig::_cfgSig.lambda [0] / (2 * CV_PI*NA_g))*(ophSig::_cfgSig.lambda[0] / (2 * CV_PI*NA_g))))*((y.at<float>(i, j)*y.at<float>(i, j))));
		}
	}
	cv::Mat G(2, size, CV_32FC2, cv::Scalar(0));
	exp(expSource2, G);

	cv::Mat G1(2, size, CV_32FC2, cv::Scalar(0));
	fftshift(G, G1);
	cv::Mat OUT_H(2, size, CV_32FC2, cv::Scalar(0));
	fft2d(buffer, OUT_H);
	cv::Mat mid(2, size, CV_32FC2, cv::Scalar(0));
	mul(G1, F1, mid);
	cv::Mat HPO(2, size, CV_32FC2, cv::Scalar(0));
	mul(mid, OUT_H, HPO);
	ifft2d(HPO, OUT_H);
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			ophSig::complexH.at<std::complex<float>>(i, j, 0)._Val[0] = OUT_H.at<std::complex<float>>(i, j).real();
			ophSig::complexH.at<std::complex<float>>(i, j, 0)._Val[1] = OUT_H.at<std::complex<float>>(i, j).imag();
		}
	}
	return true;
}

bool ophSigConvert::sigConvertCAC() {
	int size[] = { ophSig::_cfgSig.rows,ophSig::_cfgSig.cols };
	int size1[] = { ophSig::_cfgSig.rows,ophSig::_cfgSig.cols,3 };
	cv::Mat H1(2, size, CV_32FC2, cv::Scalar(0));
	cv::Mat OUT_H(3, size1, CV_32FC2, cv::Scalar(0));
	for (int z = 0; z < 3; z++)
	{
		for (int i = 0; i < ophSig::_cfgSig.rows; i++)
		{
			for (int j = 0; j < ophSig::_cfgSig.cols; j++)
			{
				H1.at<std::complex<float>>(i, j) = ophSig::complexH.at<std::complex<float>>(i, j, z);
			}
		}
		float sigmaf = ((_foc[2] - _foc[z])*ophSig::_cfgSig.lambda[z]) / (4 * CV_PI);
	
		cv::Mat r(1, ophSig::_cfgSig.rows, CV_32FC1, cv::Scalar(0));
		cv::Mat c(1, ophSig::_cfgSig.cols, CV_32FC1, cv::Scalar(0));
		r = linspace(1, ophSig::_cfgSig.rows, ophSig::_cfgSig.rows);
		c = linspace(1, ophSig::_cfgSig.cols, ophSig::_cfgSig.cols);
		cv::Mat X(1, ophSig::_cfgSig.rows, CV_32FC1, cv::Scalar(0));
		cv::Mat Y(1, ophSig::_cfgSig.cols, CV_32FC1, cv::Scalar(0));
		for (int i = 0; i < ophSig::_cfgSig.rows; i++)
		{
			X.at<float>(i) = (2 * CV_PI*(r.at<float>(i) - 1) / _radius - CV_PI*(ophSig::_cfgSig.rows - 1) / _radius);
		}

		for (int i = 0; i < ophSig::_cfgSig.cols; i++)
		{
			Y.at<float>(i) = (2 * CV_PI*(c.at<float>(i) - 1) / _radius - CV_PI*(ophSig::_cfgSig.cols - 1) / _radius);
		}
		cv::Mat x(2, size, CV_32FC1, cv::Scalar(0));
		cv::Mat y(2, size, CV_32FC1, cv::Scalar(0));
		meshgrid(X, Y, x, y);
		cv::Mat FFZP(2, size, CV_32FC2, cv::Scalar(0));
		for (int i = 0; i < ophSig::_cfgSig.rows; i++)
		{
			for (int j = 0; j < ophSig::_cfgSig.cols; j++)
			{
				FFZP.at<std::complex<float>>(i, j)._Val[1] = sigmaf*((x.at<float>(i, j)*x.at<float>(i, j)) + (y.at<float>(i, j)*y.at<float>(i, j)));
			}
		}
		exp(FFZP, FFZP);
		cv::Mat FFZP1(2, size, CV_32FC2, cv::Scalar(0));
		fftshift(FFZP, FFZP1);
		cv::Mat FH(2, size, CV_32FC2, cv::Scalar(0));
		fft2d(H1, FH);
		cv::Mat con(2, size, CV_32FC2, cv::Scalar(0));
		conj(FFZP1, con);
		cv::Mat FH_CAC(2, size, CV_32FC2, cv::Scalar(0));
		mul(FH, con, FH_CAC);
		cv::Mat IFH_CAC(2, size, CV_32FC2, cv::Scalar(0));
		ifft2d(FH_CAC, IFH_CAC);
		for (int i = 0; i < ophSig::_cfgSig.rows; i++)
		{
			for (int j = 0; j < ophSig::_cfgSig.cols; j++)
			{
				ophSig::complexH.at<std::complex<float>>(i, j, z) = IFH_CAC.at<std::complex<float>>(i, j);
			}
		}
	}
	return true;
}

void ophSigConvert::ophFree(void) {

}