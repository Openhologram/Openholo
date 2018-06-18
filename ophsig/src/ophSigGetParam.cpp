#include "ophSigGetParam.h"

ophSigGetParam::ophSigGetParam(void) {

}

bool ophSigGetParam::loadParam(std::string cfg) {

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
			else if (strcmp(title.c_str(), "depth_range_of_Max_value") == 0) {
				_zMax = stof(value);
			}
			else if (strcmp(title.c_str(), "depth_range_of_Min_value") == 0) {
				_zMin = stof(value);
			}
			else if (strcmp(title.c_str(), "sampling_count") == 0) {
				_sampN = stoi(value);
			}
			LineStream.clear();
		}
	}

	inFile.close();
	return true;
}

float ophSigGetParam::sigGetParamSF(float zMax, float zMin, int sampN) {

	cv::Mat H_temp;
	float dz = (zMax - zMin) / sampN;
	cv::Mat F = linspace(1, sampN, sampN + 1);
	cv::Mat z = linspace(1, sampN, sampN + 1);

	cv::Mat temp;
	cv::Mat OUT_H;
	float max = 0;
	int index = 0;
	float Z;

	cv::Mat H(complexH.size[0], complexH.size[1], CV_32FC2, cv::Scalar(0));
	for (int i = 0; i < H.rows; i++)
	{
		for (int j = 0; j < H.cols; j++)
		{
			H.at<std::complex<float>>(i, j) = complexH.at<std::complex<float>>(i, j, 0);
		}
	}



	int size[] = { H.rows, H.cols };
	cv::Mat I(2, size, CV_32FC2, cv::Scalar(0));

	for (int n = 0; n < sampN + 1; n++)
	{
		cv::Mat F_l(2, size, CV_32FC2, cv::Scalar(0));
		F.at<float>(n) = 0;

		z.at<float>(n) = -((n)* dz + zMin);

		I = ophSig::propagationHolo(H, z.at<float>(n));


		for (int i = 0; i < I.rows - 2; i++)
		{
			for (int j = 0; j < I.cols - 2; j++)
			{
				if (std::abs(I.at<std::complex<float>>(i + 2, j)._Val[0] - I.at<std::complex<float>>(i, j)._Val[0]) >= 0.1)
				{
					F_l.at<std::complex<float>>(i, j)._Val[0] = std::abs(I.at<std::complex<float>>(i + 2, j)._Val[0] - I.at<std::complex<float>>(i, j)._Val[0]) * std::abs(I.at<std::complex<float>>(i + 2, j)._Val[0] - I.at<std::complex<float>>(i, j)._Val[0]);
				}
				else if (std::abs(I.at<std::complex<float>>(i, j + 2)._Val[0] - I.at<std::complex<float>>(i, j)._Val[0]) >= 0.1)
				{
					F_l.at<std::complex<float>>(i, j)._Val[0] = std::abs(I.at<std::complex<float>>(i, j + 2)._Val[0] - I.at<std::complex<float>>(i, j)._Val[0]) * std::abs(I.at<std::complex<float>>(i, j + 2)._Val[0] - I.at<std::complex<float>>(i, j)._Val[0]);
				}
				F.at<float>(n) += F_l.at<std::complex<float>>(i, j)._Val[0];
			}
		}


		F.at<float>(n) = -F.at<float>(n);


	}

	ophSig::max(F, max);
	for (int i = 0; i <= F.rows; i++)
	{
		if (F.at<float>(i) == max)
		{
			index = i;
			break;
		}
	}
	return -z.at<float>(index, 0);
}

float ophSigGetParam::sigGetParamAT() {

	cv::Mat r, c;
	cv::Mat kx, ky;
	cv::Mat G;
	cv::Mat H_temp;

	cv::Mat t, tn;
	cv::Mat yn;
	cv::Mat yn1;
	cv::Mat Ab_yn, Ab_yn_half;


	float max = 0;
	int index = 0;

	cv::Mat H(complexH.size[0], complexH.size[1], CV_32FC2, cv::Scalar(0));
	for (int i = 0; i < H.rows; i++)
	{
		for (int j = 0; j < H.cols; j++)
		{
			H.at<std::complex<float>>(i, j) = complexH.at<std::complex<float>>(i, j, 0);
		}
	}


	int size[] = { H.size[0], H.size[1] };

	cv::Mat Hsyn(2, size, CV_32FC2, cv::Scalar(0));
	cv::Mat Flr(2, size, CV_32FC2, cv::Scalar(0));
	cv::Mat Fli(2, size, CV_32FC2, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat temp2(2, size, CV_32FC2, cv::Scalar(0));
	cv::Mat Fo(2, size, CV_32FC2, cv::Scalar(0));
	cv::Mat Fo1(2, size, CV_32FC2, cv::Scalar(0));
	cv::Mat Fon;



	r = ophSig::linspace(1, _cfgSig.rows, _cfgSig.rows);
	c = ophSig::linspace(1, _cfgSig.cols, _cfgSig.cols);

	for (int i = 0; i < r.rows; i++)
	{
		r.at<float>(i) = (2 * CV_PI*(r.at<float>(i) - 1) / _cfgSig.width - CV_PI*(_cfgSig.rows - 1) / _cfgSig.width);
	}

	for (int i = 0; i < c.rows; i++)
	{
		c.at<float>(i) = (2 * CV_PI*(c.at<float>(i) - 1) / _cfgSig.height - CV_PI*(_cfgSig.cols - 1) / _cfgSig.height);
	}
	ophSig::meshgrid(r, c, kx, ky);


	float NA_g = 0.025;


	temp.create(kx.rows, kx.cols, CV_32FC1);
	ophSig::mul(kx, kx, kx);
	ophSig::mul(ky, ky, ky);
	ophSig::add(kx, ky, temp);

	temp = (-CV_PI * (_cfgSig.lambda[0] / (2 * CV_PI * NA_g)) * (_cfgSig.lambda[0] / (2 * CV_PI * NA_g)) * temp);

	exp(temp, G);

	for (int i = 0; i < H.size[0]; i++)
	{
		for (int j = 0; j <H.size[1]; j++)
		{
			Flr.at<std::complex<float>>(i, j)._Val[0] = H.at<std::complex<float>>(i, j)._Val[0];
			Fli.at<std::complex<float>>(i, j)._Val[0] = H.at<std::complex<float>>(i, j)._Val[1];
		}
	}

	ophSig::fft2d(Flr, Flr);
	ophSig::fft2d(Fli, Fli);
	for (int i = 0; i < Flr.size[0]; i++)
	{
		for (int j = 0; j < Flr.size[1]; j++)
		{
			Flr.at<std::complex<float>>(i, j)._Val[1] = 0;
			Fli.at<std::complex<float>>(i, j)._Val[1] = 0;
		}
	}

	for (int i = 0; i < Hsyn.size[0]; i++)
	{
		for (int j = 0; j < Hsyn.size[1]; j++)
		{
			Hsyn.at<std::complex<float>>(i, j)._Val[0] = Flr.at<std::complex<float>>(i, j)._Val[0];
			Hsyn.at<std::complex<float>>(i, j)._Val[1] = Fli.at<std::complex<float>>(i, j)._Val[0];
		}
	}

	temp.create(2, size, CV_32FC2);

	ophSig::mul(Hsyn, G, Hsyn);
	ophSig::mul(Hsyn, Hsyn, temp);


	ophSig::abs(Hsyn, temp2);



	ophSig::mul(temp2, temp2, temp2);

	for (int i = 0; i < temp2.rows; i++)
	{
		for (int j = 0; j < temp2.cols; j++)
		{
			temp2.at<std::complex<float>>(i, j) += pow(10, -300);
		}
	}

	ophSig::div(temp, temp2, Fo);
	ophSig::fftshift2d(Fo, Fo1);
	t = ophSig::linspace(0, 1, _cfgSig.rows / 2 + 1);
	tn.create(t.rows, t.cols, CV_32FC1);

	for (int i = 0; i < tn.size[0]; i++)
	{
		tn.at<float>(i) = pow(t.at<float>(i), 0.5);
	}
	Fon.create(Fo.size[0] / 2 + 1, 1, CV_32FC1);

	for (int i = 0; i < Hsyn.size[0]; i++)
	{
		for (int j = 0; j < Hsyn.size[1]; j++)
		{
			Fo1.at<std::complex<float>>(i, j)._Val[1] = 0;
		}
	}
	for (int i = 0; i < Fo.size[0] / 2 + 1; i++)
	{
		Fon.at<float>(i) = Fo1.at<std::complex<float>>(_cfgSig.rows / 2 - 1, _cfgSig.rows / 2 - 1 + i)._Val[0];
	}
	yn.create(tn.rows, tn.cols, CV_32FC1);
	yn1.create(tn.rows, tn.cols, CV_32FC2);
	ophSig::linInterp(t, Fon, tn, yn);

	for (int i = 0; i < yn.rows; i++)
	{
		for (int j = 0; j < yn.cols; j++)
		{
			yn1.at<std::complex<float>>(i, j) = yn.at<float>(i, j);
		}
	}
	ophSig::fft1d(yn1, yn1);


	Ab_yn.create(yn.rows, yn.cols, CV_32FC2);
	ophSig::abs(yn1, Ab_yn);




	Ab_yn_half.create(_cfgSig.rows / 4 + 1, 1, CV_32FC2);


	for (int i = 0; i < _cfgSig.rows / 4 + 1; i++)
	{
		Ab_yn_half.at<std::complex<float>>(i) = Ab_yn.at<std::complex<float>>(_cfgSig.rows / 4 + i - 1);
	}

	ophSig::max(Ab_yn_half, max);

	for (int i = 0; i < Ab_yn_half.size[0]; i++)
	{
		if (Ab_yn_half.at<std::complex<float>>(i)._Val[0] == max)
		{
			index = i;
			break;
		}
	}
	return index;
}

void ophSigGetParam::ophFree(void) {

}