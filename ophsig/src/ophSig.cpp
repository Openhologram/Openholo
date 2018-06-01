#include "ophSig.h"


ophSig::ophSig(void) {

}

cv::Mat ophSig::linspace(float first, float last, int len) {
	cv::Mat result(len, 1, CV_32FC1, cv::Scalar(0));
	float step = (last - first) / (len - 1);
	for (int i = 0; i < len; i++) { result.at<float>(i) = first + i*step; }
	return result;
}
void ophSig::add(cv::Mat &A, cv::Mat &B, cv::Mat &out)
{
	if ((A.channels() == 1) && (B.channels() == 1) == 1)	
	{
		if ((A.dims == 2) && (B.dims == 2) == 1)
		{
			for (int row = 0; row < A.size[0]; row++)
			{
				for (int col = 0; col < A.size[1]; col++)
				{
					*(out.ptr<float>(row, col)) = *(A.ptr<float>(row, col)) + *(B.ptr<float>(row, col));
				}
			}
		}
	}
}

void ophSig::fftshift2d(cv::Mat &in, cv::Mat &out)
{
	if (in.channels() == 1)

	{
		int xshift = in.size[0] / 2;
		int yshift = in.size[1] / 2;
		for (int i = 0; i < in.size[0]; i++)
		{
			int ii = (i + xshift) % in.size[0];
			for (int j = 0; j < in.size[1]; j++)
			{
				int jj = (j + yshift) % in.size[1];
				(*(out.ptr<float>(ii, jj))) = (*(in.ptr<float>(i, j)));


			}

		}
	}
	else
	{
		int xshift = in.size[0] / 2;
		int yshift = in.size[1] / 2;
		for (int i = 0; i < in.size[0]; i++)
		{
			int ii = (i + xshift) % in.size[0];
			for (int j = 0; j < in.size[1]; j++)
			{
				int jj = (j + yshift) % in.size[1];
				(*(out.ptr<std::complex<float>>(ii, jj)))._Val[0] = (*(in.ptr<std::complex<float>>(i, j))).real();
				(*(out.ptr<std::complex<float>>(ii, jj)))._Val[1] = (*(in.ptr<std::complex<float>>(i, j))).imag();

			}

		}
	}
}

void ophSig::fft1d(cv::Mat &in, cv::Mat &out)
{
	CWO fft;
	cwoComplex *input = new cwoComplex[in.size[0]];
	cwoComplex *output = new cwoComplex[in.size[0]];
	for (int i = 0; i < in.size[0]; i++)
	{
		input[i].cplx[0] = (float)(*(in.ptr<std::complex<float>>(i))).real();
		input[i].cplx[1] = (float)(*(in.ptr<std::complex<float>>(i))).imag();

	}
	fft.Create(in.size[0]);

	fft.__FFT(input, output, 0);

	for (int i = 0; i < in.size[0]; i++)
	{
		(*(out.ptr<std::complex<float>>(i)))._Val[0] = (float)output[i].cplx[0];
		(*(out.ptr<std::complex<float>>(i)))._Val[1] = (float)output[i].cplx[1];
	}
	delete[]input;
	delete[]output;
}


void ophSig::linInterp(cv::Mat &X, cv::Mat &in, cv::Mat &Xq, cv::Mat &out)
{
	if (in.channels() == 1)
	{
		int size = X.size[0]; 
		int i = 0;
		for (int j = 0; j < out.size[0]; j++)
		{
			if ((*(Xq.ptr < float >(j))) >= (*(X.ptr <float>(size - 2))))
			{															
				i = size - 2;
			}
			else
			{
				while ((*(Xq.ptr <float>(j))) >(*(X.ptr <float>(i + 1)))) i++;
			}

			float xL = (*(X.ptr <float>(i)));
			float yL = (*(in.ptr <float>(i)));
			float xR = (*(X.ptr <float>(i + 1)));
			float yR = (*(in.ptr <float>(i + 1)));

			double dydx = (yR - yL) / (xR - xL);                                  

			(*(out.ptr <float>(j))) = yL + dydx * ((*(Xq.ptr <float>(j))) - xL);


		}
	}
	else
	{
		int size = X.size[0];
		int i = 0;
		for (int j = 0; j < out.size[0]; j++)
		{
			if ((*(Xq.ptr<float>(j))) >= (*(X.ptr<float>(size - 2))))                                              
			{																			
				i = size - 2;
			}
			else
			{
				while ((*(Xq.ptr<float>(j))) >(*(X.ptr<float>(i + 1)))) i++;  
			}

			float xL = (*(X.ptr<float>(i)));
			float yL = (*(in.ptr<std::complex<float>>(i))).real();
			float xR = (*(X.ptr<float>(i + 1)));
			float yR = (*(in.ptr<std::complex<float>>(i + 1))).real();

			float iyL = (*(in.ptr<std::complex<float>>(i))).imag();

			float iyR = (*(in.ptr<std::complex<float>>(i + 1))).imag();

			float dydx = (yR - yL) / (xR - xL);
			double idydx = (iyR - iyL) / (xR - xL);
			(*(out.ptr<std::complex<float>>(j)))._Val[0] = yL + dydx * ((*(Xq.ptr<float>(j))) - xL);
			(*(out.ptr<std::complex<float>>(j)))._Val[1] = iyL + idydx * ((*(Xq.ptr<float>(j))) - xL);

		}
	}
}

void ophSig::div(cv::Mat &A, cv::Mat &B, cv::Mat &out)
{
	if ((A.channels() == 2) && (B.channels() == 2) == 1)		
	{
		if ((A.dims == 2) && (B.dims == 2) == 1)		
		{
			for (int row = 0; row < A.size[0]; row++)
			{
				for (int col = 0; col < A.size[1]; col++)
				{
					*(out.ptr<std::complex<float>>(row, col)) = *(A.ptr<std::complex<float>>(row, col)) / *(B.ptr<std::complex<float>>(row, col));		
				}
			}
		}
	}
}

void ophSig::exp(cv::Mat &in, cv::Mat &out)
{
	if ((in.channels() == 1) == 1)	
	{
		if (in.dims == 2)
		{
			int size[] = { in.rows, in.cols };
			out.create(2, size, CV_32FC1);
			for (int row = 0; row < in.rows; row++)
			{
				for (int col = 0; col < in.cols; col++)
				{
					*(out.ptr<float>(row, col)) = std::exp(*(in.ptr<float>(row, col)));
				}
			}
		}
	}
	else if ((in.channels() == 2) == 1)
	{
		int size[] = { in.rows, in.cols };
		out.create(2, size, CV_32FC2);
		for (int row = 0; row < in.rows; row++)
		{
			for (int col = 0; col < in.cols; col++)
			{
				*(out.ptr<std::complex<float>>(row, col)) = std::exp(*(in.ptr<std::complex<float>>(row, col)));
			}
		}
	}
}

void ophSig::mul(cv::Mat &A, cv::Mat &B, cv::Mat &out)
{
	if ((A.channels() == 1) && (B.channels() == 1) == 1)	
	{
		if ((A.dims == 2) && (B.dims == 2) == 1)
		{
			for (int row = 0; row < A.size[0]; row++)
			{
				for (int col = 0; col < A.size[1]; col++)
				{
					*(out.ptr<float>(row, col)) = *(A.ptr<float>(row, col)) * *(B.ptr<float>(row, col));
				}
			}
		}
	}
	else if ((A.channels() == 2) && (B.channels() == 2) == 1)		
	{
		if ((A.dims == 2) && (B.dims == 2) == 1)		
		{
			for (int row = 0; row < A.size[0]; row++)
			{
				for (int col = 0; col < A.size[1]; col++)
				{
					*(out.ptr<std::complex<float>>(row, col)) = *(A.ptr<std::complex<float>>(row, col)) * *(B.ptr<std::complex<float>>(row, col));		

				}
			}
		}
	}
}

void ophSig::min(cv::Mat &in, float &out)
{
	out = in.at<float>(0, 0);
	for (int i = 0; i < in.size[0]; i++)
	{
		for (int j = 0; j < in.size[1]; j++)
		{

			if (in.at<float>(i, j) < out) { out = in.at<float>(i, j); }
		}
	}
}

void ophSig::max(cv::Mat &in, float &out)
{
	out = in.at<float>(0, 0);
	for (int i = 0; i < in.size[0]; i++)
	{
		for (int j = 0; j < in.size[1]; j++)
		{

			if (in.at<float>(i, j) > out) { out = in.at<float>(i, j); }
		}
	}
}

void  ophSig::fftshift(cv::Mat &in, cv::Mat &out)
{
	if (in.channels() == 1)
	{
		int xshift = in.size[0] / 2;
		int yshift = in.size[1] / 2;
		for (int i = 0; i < in.size[0]; i++)
		{
			int ii = (i + xshift) % in.size[0];
			for (int j = 0; j < in.size[1]; j++)
			{
				int jj = (j + yshift) % in.size[1];
				(*(out.ptr<float>(ii, jj))) = (*(in.ptr<float>(i, j)));
			}
		}
	}
	else
	{
		int xshift = in.size[0] / 2;
		int yshift = in.size[1] / 2;
		for (int i = 0; i < in.size[0]; i++)
		{
			int ii = (i + xshift) % in.size[0];
			for (int j = 0; j < in.size[1]; j++)
			{
				int jj = (j + yshift) % in.size[1];
				(*(out.ptr<std::complex<float>>(ii, jj)))._Val[0] = (*(in.ptr<std::complex<float>>(i, j))).real();
				(*(out.ptr<std::complex<float>>(ii, jj)))._Val[1] = (*(in.ptr<std::complex<float>>(i, j))).imag();

			}

		}
	}
}

void ophSig::fft2d(cv::Mat &in, cv::Mat &out)
{
	CWO cwo;
	cwoComplex *input = new cwoComplex[in.size[0] * in.size[1]];
	cwoComplex *output = new cwoComplex[in.size[0] * in.size[1]];
	for (int i = 0; i < in.size[0]; i++)
	{
		for (int j = 0; j < in.size[1]; j++)
		{
			input[i*in.size[0] + j].cplx[0] = (float)(*(in.ptr<std::complex<float>>(i, j))).real();
			input[i*in.size[0] + j].cplx[1] = (float)(*(in.ptr<std::complex<float>>(i, j))).imag();
		}
	}

	cwo.Create(in.size[0], in.size[1]);
	cwo.__FFT(input, output, 0);

	for (int i = 0; i < in.size[0]; i++)
	{
		for (int j = 0; j < in.size[1]; j++)
		{
			(*(out.ptr<std::complex<float>>(i, j)))._Val[0] = (float)output[i*in.size[0] + j].cplx[0];
			(*(out.ptr<std::complex<float>>(i, j)))._Val[1] = (float)output[i*in.size[0] + j].cplx[1];
		}

	}
	delete[] input;
	delete[] output;
}

void ophSig::ifft2d(cv::Mat &in, cv::Mat &out)
{
	CWO cwo;
	cwoComplex *input = new cwoComplex[in.size[0] * in.size[1]];
	cwoComplex *output = new cwoComplex[in.size[0] * in.size[1]];
	for (int i = 0; i < in.size[0]; i++)
	{
		for (int j = 0; j < in.size[1]; j++)
		{
			input[i*in.size[0] + j].cplx[0] = (float)(*(in.ptr<std::complex<float>>(i, j))).real();
			input[i*in.size[0] + j].cplx[1] = (float)(*(in.ptr<std::complex<float>>(i, j))).imag();
		}
	}
	cwo.Create(in.size[0], in.size[1]);
	cwo.__IFFT(input, output);

	for (int i = 0; i < in.size[0]; i++)
	{
		for (int j = 0; j < in.size[1]; j++)
		{
			(*(out.ptr<std::complex<float>>(i, j)))._Val[0] = (float)(output[i*in.size[0] + j].cplx[0] / (in.size[0] * in.size[1]));
			(*(out.ptr<std::complex<float>>(i, j)))._Val[1] = (float)(output[i*in.size[0] + j].cplx[1] / (in.size[0] * in.size[1]));
		}

	}
	delete[] input;
	delete[] output;
}

void ophSig::mean(cv::Mat &in, float &out)
{
	out = 0;
	for (int i = 0; i < in.size[0]; i++)
	{
		for (int j = 0; j < in.size[1]; j++)
		{
			out += in.at<float>(i, j);
		}
	}
	out = out / (in.size[0] * in.size[1]);
}
void ophSig::conj(cv::Mat &in, cv::Mat &out)
{
	for (int row = 0; row < in.rows; row++)
	{
		for (int col = 0; col < in.cols; col++)
		{
			*(out.ptr<std::complex<float>>(row, col)) = std::conj(*(in.ptr<std::complex<float>>(row, col)));
		}
	}
}

void ophSig::meshgrid(const cv::Mat&x, const cv::Mat &y, cv::Mat &a, cv::Mat &b)
{
	int array[2];
	array[0] = (int)x.total();
	array[1] = (int)y.total();

	if ((x.channels() == 2) == 1)
	{
		a.create(2, array, CV_32FC2);
		b.create(2, array, CV_32FC2);
		cv::repeat(x.reshape(1, 1), y.total(), 1, a);
		cv::repeat(y.reshape(1, 1).t(), 1, x.total(), b);
	}
	else if ((x.channels() == 1) == 1)
	{
		a.create(2, array, CV_32FC1);
		b.create(2, array, CV_32FC1);
		cv::repeat(x.reshape(1, 1), y.total(), 1, a);
		cv::repeat(y.reshape(1, 1).t(), 1, x.total(), b);
	}
}

void ophSig::abs(cv::Mat &in, cv::Mat &out)
{
	if ((in.channels() == 2) == 1)		
	{
		int size[] = { in.rows, in.cols };
		out.create(2, size, CV_32FC2);
		for (int row = 0; row < in.rows; row++)
		{
			for (int col = 0; col < in.cols; col++)
			{
				*(out.ptr<std::complex<float>>(row, col)) = std::abs(*(in.ptr<std::complex<float>>(row, col)));
			}
		}
	}
	else if ((in.channels() == 1) == 1)
	{
		for (int row = 0; row < in.rows; row++)
		{
			for (int col = 0; col < in.cols; col++)
			{
				*(out.ptr<float>(row, col)) = std::abs(*(in.ptr<float>(row, col)));
			}
		}
	}
}
bool ophSig::loadHolo(std::string cosh, std::string sinh, std::string type,float flag) {
	if (type == "bmp")
	{
		if (flag == 1)
		{
			cv::Mat cos = cv::imread(cosh, CV_LOAD_IMAGE_COLOR);
			cv::Mat sin = cv::imread(sinh, CV_LOAD_IMAGE_COLOR);
			int size[] = { _cfgSig.rows,_cfgSig.cols,3 };
			complexH.create(3, size, CV_32FC2);
			cv::Mat cosh_H(3, size, CV_32F, cv::Scalar(0));
			cv::Mat sinh_H(3, size, CV_32F, cv::Scalar(0));
			for (int i = 0; i < _cfgSig.rows; i++)
			{
				for (int j = 0; j < _cfgSig.cols; j++)
				{
					cosh_H.at<float>(i, j, 0) = (float)cos.at<cv::Vec3b>(i, j)[0];
					cosh_H.at<float>(i, j, 1) = (float)cos.at<cv::Vec3b>(i, j)[1];
					cosh_H.at<float>(i, j, 2) = (float)cos.at<cv::Vec3b>(i, j)[2];
					sinh_H.at<float>(i, j, 0) = (float)sin.at<cv::Vec3b>(i, j)[0];
					sinh_H.at<float>(i, j, 1) = (float)sin.at<cv::Vec3b>(i, j)[1];
					sinh_H.at<float>(i, j, 2) = (float)sin.at<cv::Vec3b>(i, j)[2];

				}
			}
			cv::Mat cosh_h(2, size, CV_32F, cv::Scalar(0));
			cv::Mat sinh_h(2, size, CV_32F, cv::Scalar(0));
			for (int z = 0; z < 3; z++)
			{
				for (int i = 0; i < _cfgSig.rows; i++)
				{
					for (int j = 0; j < _cfgSig.cols; j++)
					{
						cosh_h.at<float>(i, j) = cosh_H.at<float>(i, j, z);
						sinh_h.at<float>(i, j) = sinh_H.at<float>(i, j, z);
					}
				}
				float out;
				mean(sinh_h, out);
				sinh_h = sinh_h / out;
				mean(cosh_h, out);
				cosh_h = cosh_h / out;
				cv::Mat cosh_abs(2, size, CV_32F, cv::Scalar(0));
				cv::Mat sinh_abs(2, size, CV_32F, cv::Scalar(0));
				abs(sinh_h, sinh_abs);
				abs(cosh_h, cosh_abs);
				max(sinh_abs, out);
				sinh_h = sinh_h / out;
				max(cosh_abs, out);
				cosh_h = cosh_h / out;
				min(sinh_h, out);
				for (int i = 0; i < _cfgSig.rows; i++)
				{
					for (int j = 0; j < _cfgSig.cols; j++)
					{
						sinh_h.at<float>(i, j) = sinh_h.at<float>(i, j) - out;
					}
				}

				min(cosh_h, out);
				for (int i = 0; i < _cfgSig.rows; i++)
				{
					for (int j = 0; j < _cfgSig.cols; j++)
					{
						cosh_h.at<float>(i, j) = cosh_h.at<float>(i, j) - out;
					}
				}

				for (int i = 0; i < cosh_H.size[0]; i++)
				{
					for (int j = 0; j < cosh_H.size[1]; j++)
					{
						complexH.at<std::complex<float>>(i, j, z)._Val[0] = cosh_h.at<float>(i, j);
						complexH.at<std::complex<float>>(i, j, z)._Val[1] = sinh_h.at<float>(i, j);
					}
				}
			}

		}
		else if (flag == 0)
		{
			cv::Mat cos = cv::imread(cosh, 0);
			cv::Mat sin = cv::imread(sinh, 0);
			int size1[] = { cos.size[0],cos.size[1],1 };
			int size[] = { cos.size[0],cos.size[1] };
			complexH.create(3, size1, CV_32FC2);
			cv::Mat cosh_h(2, size, CV_32F, cv::Scalar(0));
			cv::Mat sinh_h(2, size, CV_32F, cv::Scalar(0));
			for (int i = 0; i < cos.size[0]; i++)
			{
				for (int j = 0; j < cos.size[1]; j++)
				{
					cosh_h.at<float>(i, j) = (float)cos.at<uchar>(i, j);
					sinh_h.at<float>(i, j) = (float)sin.at<uchar>(i, j);

				}
			}
			float out;
			mean(sinh_h, out);
			sinh_h = sinh_h / out;
			mean(cosh_h, out);
			cosh_h = cosh_h / out;
			cv::Mat cosh_abs(2, size, CV_32F, cv::Scalar(0));
			cv::Mat sinh_abs(2, size, CV_32F, cv::Scalar(0));
			abs(sinh_h, sinh_abs);
			abs(cosh_h, cosh_abs);
			max(sinh_abs, out);
			sinh_h = sinh_h / out;
			max(cosh_abs, out);
			cosh_h = cosh_h / out;
			min(sinh_h, out);
			for (int i = 0; i < _cfgSig.rows; i++)
			{
				for (int j = 0; j < _cfgSig.cols; j++)
				{
					sinh_h.at<float>(i, j) = sinh_h.at<float>(i, j) - out;
				}
			}

			min(cosh_h, out);
			for (int i = 0; i < _cfgSig.rows; i++)
			{
				for (int j = 0; j < _cfgSig.cols; j++)
				{
					cosh_h.at<float>(i, j) = cosh_h.at<float>(i, j) - out;
				}
			}

			for (int i = 0; i < _cfgSig.rows; i++)
			{
				for (int j = 0; j < _cfgSig.cols; j++)
				{
					complexH.at<std::complex<float>>(i, j, 0)._Val[0] = cosh_h.at<float>(i, j);
					complexH.at<std::complex<float>>(i, j, 0)._Val[1] = sinh_h.at<float>(i, j);
				}
			}
		}

	}

	else if (type == "bin")
	{
		if (flag == 0)
		{
			std::ifstream fileStream(cosh, std::ifstream::binary);
			std::ifstream fileStream2(sinh, std::ifstream::binary);
			int dim[] = { _cfgSig.rows,_cfgSig.cols };
			int dim1[] = { _cfgSig.rows,_cfgSig.cols,1 };
			complexH.create(3, dim1, CV_32FC2);
			cv::Mat cosh_h(2, dim, CV_32F, cv::Scalar(0));
			cv::Mat sinh_h(2, dim, CV_32F, cv::Scalar(0));
			float total = _cfgSig.rows*_cfgSig.cols;
			float *temp1 = new  float[total];
			float *temp2 = new  float[total];
			int i = 0;
			fileStream.read(reinterpret_cast<char*>(temp1), sizeof(float) * total);
			fileStream2.read(reinterpret_cast<char*>(temp2), sizeof(float) * total);

			for (int row = 0; row < _cfgSig.rows; row++)
			{
				for (int col = 0; col < _cfgSig.cols; col++)
				{
					cosh_h.at<float>(col, row) = temp1[i];
					sinh_h.at<float>(col, row) = temp2[i];
					i++;
				}
			}
			float out;
			mean(sinh_h, out);
			sinh_h = sinh_h / out;
			mean(cosh_h, out);
			cosh_h = cosh_h / out;
			cv::Mat cosh_abs(2, dim, CV_32F, cv::Scalar(0));
			cv::Mat sinh_abs(2, dim, CV_32F, cv::Scalar(0));
			abs(sinh_h, sinh_abs);
			abs(cosh_h, cosh_abs);
			max(sinh_abs, out);
			sinh_h = sinh_h / out;
			max(cosh_abs, out);
			cosh_h = cosh_h / out;
			min(sinh_h, out);
			for (int i = 0; i < _cfgSig.rows; i++)
			{
				for (int j = 0; j < _cfgSig.cols; j++)
				{
					sinh_h.at<float>(i, j) = sinh_h.at<float>(i, j) - out;
				}
			}

			min(cosh_h, out);
			for (int i = 0; i < _cfgSig.rows; i++)
			{
				for (int j = 0; j < _cfgSig.cols; j++)
				{
					cosh_h.at<float>(i, j) = cosh_h.at<float>(i, j) - out;
				}
			}

			for (int i = 0; i < cosh_h.size[0]; i++)
			{
				for (int j = 0; j < cosh_h.size[1]; j++)
				{
					complexH.at<std::complex<float>>(i, j, 0)._Val[0] = cosh_h.at<float>(i, j);
					complexH.at<std::complex<float>>(i, j, 0)._Val[1] = sinh_h.at<float>(i, j);
				}
			}
			fileStream.close();
			fileStream2.close();
			delete[]temp1;
			delete[]temp2;
		}
		else if (flag == 1)
		{

			int dim[] = { _cfgSig.rows,_cfgSig.cols };
			int size[] = { _cfgSig.rows,_cfgSig.cols,3 };
			complexH.create(3, size, CV_32FC2);
			float total = _cfgSig.rows*_cfgSig.cols;
			cv::String rstTemp1 = "";
			cv::String rstTemp2 = "";
			cv::String rstTemp3 = "";
			cv::String rstTemp4 = "";
			cv::String rstTemp5 = "";
			cv::String rstTemp6 = "";

			int ret = cosh.rfind(".");
			if (cosh.rfind(".") == cv::String::npos) {
				rstTemp1 = cosh + "_R." + "bin";
				rstTemp2 = cosh + "_G." + "bin";
				rstTemp3 = cosh + "_B." + "bin";
				rstTemp4 = sinh + "_R." + "bin";
				rstTemp5 = sinh + "_G." + "bin";
				rstTemp6 = sinh + "_B." + "bin";

			}
			else {
				rstTemp1 = cosh.substr(0, ret) + "_R" + cosh.substr(ret, cosh.size());
				rstTemp2 = cosh.substr(0, ret) + "_G" + cosh.substr(ret, cosh.size());
				rstTemp3 = cosh.substr(0, ret) + "_B" + cosh.substr(ret, cosh.size());
				rstTemp4 = sinh.substr(0, ret) + "_R" + sinh.substr(ret, sinh.size());
				rstTemp5 = sinh.substr(0, ret) + "_G" + sinh.substr(ret, sinh.size());
				rstTemp6 = sinh.substr(0, ret) + "_B" + sinh.substr(ret, sinh.size());

			}
			cv::String real_name[] = { rstTemp1,rstTemp2,rstTemp3 };
			cv::String imag_name[] = { rstTemp4,rstTemp5,rstTemp6 };


			float *temp1 = new  float[total];
			float *temp2 = new  float[total];

			for (int z = 0; z < 3; z++)
			{
				std::ifstream fileStream(real_name[z], std::ifstream::binary);
				std::ifstream fileStream2(imag_name[z], std::ifstream::binary);

				fileStream.read(reinterpret_cast<char*>(temp1), sizeof(float) * total);
				fileStream2.read(reinterpret_cast<char*>(temp2), sizeof(float) * total);
				cv::Mat cosh_h(2, dim, CV_32FC1, cv::Scalar(0));
				cv::Mat sinh_h(2, dim, CV_32FC1, cv::Scalar(0));

				int i = 0;
				for (int row = 0; row < _cfgSig.rows; row++)
				{
					for (int col = 0; col < _cfgSig.cols; col++)
					{
						cosh_h.at<float>(col, row) = temp1[i];
						sinh_h.at<float>(col, row) = temp2[i];
						i++;

					}
				}

				float out;
				mean(sinh_h, out);
				sinh_h = sinh_h / out;
				mean(cosh_h, out);
				cosh_h = cosh_h / out;
				cv::Mat cosh_abs(2, dim, CV_32F, cv::Scalar(0));
				cv::Mat sinh_abs(2, dim, CV_32F, cv::Scalar(0));
				abs(sinh_h, sinh_abs);
				abs(cosh_h, cosh_abs);
				max(sinh_abs, out);
				sinh_h = sinh_h / out;
				max(cosh_abs, out);
				cosh_h = cosh_h / out;
				min(sinh_h, out);
				for (int i = 0; i < _cfgSig.rows; i++)
				{
					for (int j = 0; j < _cfgSig.cols; j++)
					{
						sinh_h.at<float>(i, j) = sinh_h.at<float>(i, j) - out;
					}
				}

				min(cosh_h, out);
				for (int i = 0; i < _cfgSig.rows; i++)
				{
					for (int j = 0; j < _cfgSig.cols; j++)
					{
						cosh_h.at<float>(i, j) = cosh_h.at<float>(i, j) - out;
					}
				}

				for (int i = 0; i < _cfgSig.rows; i++)
				{
					for (int j = 0; j < _cfgSig.cols; j++)
					{
						complexH.at<std::complex<float>>(i, j, z)._Val[0] = cosh_h.at<float>(i, j);
						complexH.at<std::complex<float>>(i, j, z)._Val[1] = sinh_h.at<float>(i, j);

					}
				}
				fileStream.close();
				fileStream2.close();

			}
			delete[] temp1;
			delete[] temp2;
		}
	}
	return true;
}


bool ophSig::saveHolo(std::string cosh, std::string sinh, std::string type,float flag) {
	if (type == "bmp")
	{
		if (flag == 1)
		{
			int size[] = { _cfgSig.rows, _cfgSig.cols };
			cv::Mat sin(2, size, CV_8UC3, cv::Scalar(0));
			cv::Mat cos(2, size, CV_8UC3, cv::Scalar(0));
			for (int i = 0; i < _cfgSig.rows; i++)
			{
				for (int j = 0; j < _cfgSig.rows; j++)
				{
					cos.at<cv::Vec3b>(i, j)[0] = (uchar)(256 * complexH.at<std::complex<float>>(i, j, 0)._Val[0]);
					cos.at<cv::Vec3b>(i, j)[1] = (uchar)(256 * complexH.at<std::complex<float>>(i, j, 1)._Val[0]);
					cos.at<cv::Vec3b>(i, j)[2] = (uchar)(256 * complexH.at<std::complex<float>>(i, j, 2)._Val[0]);
					sin.at<cv::Vec3b>(i, j)[0] = (uchar)(256 * complexH.at<std::complex<float>>(i, j, 0)._Val[1]);
					sin.at<cv::Vec3b>(i, j)[1] = (uchar)(256 * complexH.at<std::complex<float>>(i, j, 1)._Val[1]);
					sin.at<cv::Vec3b>(i, j)[2] = (uchar)(256 * complexH.at<std::complex<float>>(i, j, 2)._Val[1]);
				}
			}
			cv::imwrite(cosh, cos);
			cv::imwrite(sinh, sin);
		}
		else if (flag == 0)
		{
			int size[] = { complexH.size[0], complexH.size[1] };
			cv::Mat sin(2, size, CV_8UC1, cv::Scalar(0));
			cv::Mat cos(2, size, CV_8UC1, cv::Scalar(0));
			for (int i = 0; i < complexH.size[0]; i++)
			{
				for (int j = 0; j < complexH.size[1]; j++)
				{
					cos.at<uchar>(i, j) = 256 * complexH.at<std::complex<float>>(i, j, 0).real();
					sin.at<uchar>(i, j) = 256 * complexH.at<std::complex<float>>(i, j, 0).imag();
				}
			}
			cv::imwrite(cosh, cos);
			cv::imwrite(sinh, sin);
		}

	}

	else if (type == "bin")
	{
		if (flag == 0)
		{
			std::ofstream cos(cosh, std::ios::binary);
			std::ofstream sin(sinh, std::ios::binary);
			int dim[] = { complexH.size[0], complexH.size[1] };

			float *temp1 = new  float[complexH.size[0] * complexH.size[1]];
			float *temp2 = new  float[complexH.size[0] * complexH.size[1]];
			int  i = 0;
			for (int row = 0; row < complexH.size[0]; row++)
			{
				for (int col = 0; col < complexH.size[1]; col++)
				{
					temp1[i] = complexH.at<std::complex<float>>(col, row, 0)._Val[0];
					temp2[i] = complexH.at<std::complex<float>>(col, row, 0)._Val[1];
					i++;
				}
			}


			cos.write(reinterpret_cast<const char*>(temp1), sizeof(float) * complexH.size[0] * complexH.size[1]);
			sin.write(reinterpret_cast<const char*>(temp2), sizeof(float) * complexH.size[0] * complexH.size[1]);

			cos.close();
			sin.close();
			delete[]temp1;
			delete[]temp2;
		}
		else if (flag == 1)
		{
			cv::String rstTemp1 = "";
			cv::String rstTemp2 = "";
			cv::String rstTemp3 = "";
			cv::String rstTemp4 = "";
			cv::String rstTemp5 = "";
			cv::String rstTemp6 = "";

			int ret = cosh.rfind(".");
			if (cosh.rfind(".") == cv::String::npos) {
				rstTemp1 = cosh + "_B." + "bin";
				rstTemp2 = cosh + "_G." + "bin";
				rstTemp3 = cosh + "_R." + "bin";
				rstTemp4 = sinh + "_B." + "bin";
				rstTemp5 = sinh + "_G." + "bin";
				rstTemp6 = sinh + "_R." + "bin";

			}
			else {
				rstTemp1 = cosh.substr(0, ret) + "_B" + cosh.substr(ret, cosh.size());
				rstTemp2 = cosh.substr(0, ret) + "_G" + cosh.substr(ret, cosh.size());
				rstTemp3 = cosh.substr(0, ret) + "_R" + cosh.substr(ret, cosh.size());
				rstTemp4 = sinh.substr(0, ret) + "_B" + sinh.substr(ret, sinh.size());
				rstTemp5 = sinh.substr(0, ret) + "_G" + sinh.substr(ret, sinh.size());
				rstTemp6 = sinh.substr(0, ret) + "_R" + sinh.substr(ret, sinh.size());

			}
			cv::String real_name[] = { rstTemp1,rstTemp2,rstTemp3 };
			cv::String imag_name[] = { rstTemp4,rstTemp5,rstTemp6 };
			int dim[] = { _cfgSig.rows, _cfgSig.cols };
			float *temp1 = new  float[_cfgSig.rows * _cfgSig.cols];
			float *temp2 = new  float[_cfgSig.rows * _cfgSig.cols];
			for (int z = 0; z < 3; z++)
			{

				std::ofstream cos(real_name[z], std::ios::binary);
				std::ofstream sin(imag_name[z], std::ios::binary);

				int  i = 0;
				for (int row = 0; row < _cfgSig.rows; row++)
				{
					for (int col = 0; col < _cfgSig.cols; col++)
					{
						temp1[i] = complexH.at<std::complex<float>>(col, row, z).real();
						temp2[i] = complexH.at<std::complex<float>>(col, row, z).imag();
						i++;
					}
				}


				cos.write(reinterpret_cast<const char*>(temp1), sizeof(float) * _cfgSig.rows * _cfgSig.cols);
				sin.write(reinterpret_cast<const char*>(temp2), sizeof(float) * _cfgSig.rows * _cfgSig.cols);


				cos.close();
				sin.close();
			}
			delete[]temp1;
			delete[]temp2;
		}
	}
	return true;
}

bool ophSig::loadParam(std::string cfg) {

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
			LineStream.clear();
		}
	}

	inFile.close();
	return true;
}

cv::Mat ophSig::propagationHolo(float depth) {
	std::cout << complexH.at<std::complex<float>>(0, 1, 0);
	cv::Mat temp(complexH.size[0], complexH.size[1], CV_32FC2, cv::Scalar(0));
	if (complexH.dims == 3)
	{
		for (int i = 0; i < temp.rows; i++)
		{
			for (int j = 0; j < temp.cols; j++)
			{
				temp.at<std::complex<float>>(i, j) = complexH.at<std::complex<float>>(i, j, 0);
			}
		}
	}
	else {
		complexH.copyTo(temp);
	}
	int index = 0;
	int Z = 0;
	float sigma;
	float sigmaf;
	cv::Mat x, y, kx, ky;
	cv::Mat temp2, temp3;
	cv::Mat H, FH, FHI;
	cv::Mat OUT_H;
	int size[] = { temp.rows, temp.cols };
	cv::Mat FFZP(2, size, CV_32FC2, cv::Scalar(0));
	cv::Mat FFZP2(2, size, CV_32FC2, cv::Scalar(0));
	sigma = CV_PI / (_cfgSig.lambda[0] * depth);
	sigmaf = (depth * _cfgSig.lambda[0]) / (4 * CV_PI);

	float row, col, color;
	row = temp.size[0];
	col = temp.size[1];
	color = temp.dims;

	int size1[] = { row };
	int size2[] = { col };
	cv::Mat r(1, size1, CV_32FC2, cv::Scalar(0));
	cv::Mat c(1, size2, CV_32FC2, cv::Scalar(0));

	c = r * 3;
	r = this->linspace(1, row, row);
	c = this->linspace(1, col, col);
	for (int i = 0; i < r.rows; i++)
	{
		r.at<float>(i) = (2 * CV_PI*(r.at<float>(i) - 1) / _cfgSig.width - CV_PI*(row - 1) / _cfgSig.width);
	}

	for (int i = 0; i < c.rows; i++)
	{
		c.at<float>(i) = (2 * CV_PI*(c.at<float>(i) - 1) / _cfgSig.height - CV_PI*(row - 1) / _cfgSig.height);
	}
	this->meshgrid(r, c, kx, ky);

	this->mul(kx, kx, kx);
	this->mul(ky, ky, ky);
	temp2.create(kx.rows, kx.cols, CV_32FC1);


	this->add(kx, ky, temp2);
	temp3.create(temp2.rows, temp2.cols, CV_32FC2);


	for (int i = 0; i < temp3.rows; i++)
	{
		for (int j = 0; j < temp3.cols; j++)
		{
			temp3.at<std::complex<float>>(i, j)._Val[0] = 0;
			temp3.at<std::complex<float>>(i, j)._Val[1] = temp2.at<float>(i, j);
		}
	}
	temp3 = temp3 * sigmaf;
	this->exp(temp3, FFZP);
	this->fftshift2d(FFZP, FFZP2);


	FH.create(temp.rows, temp.cols, CV_32FC2);
	this->fft2d(temp, FH);
	FHI.create(temp.rows, temp.cols, CV_32FC2);
	this->mul(FH, FFZP2, FHI);

	H.create(temp.rows, temp.cols, CV_32FC2);
	this->ifft2d(FHI, H);

	return H;
}

cv::Mat ophSig::propagationHolo(cv::Mat complexH, float depth) {
	std::cout << complexH.at<std::complex<float>>(0, 1, 0);
	cv::Mat temp(complexH.size[0], complexH.size[1], CV_32FC2, cv::Scalar(0));
	if (complexH.dims == 3)
	{
		for (int i = 0; i < temp.rows; i++)
		{
			for (int j = 0; j < temp.cols; j++)
			{
				temp.at<std::complex<float>>(i, j) = complexH.at<std::complex<float>>(i, j, 0);
			}
		}
	}
	else {
		complexH.copyTo(temp);
	}
	int index = 0;
	int Z = 0;
	float sigma;
	float sigmaf;
	cv::Mat x, y, kx, ky;
	cv::Mat temp2, temp3;
	cv::Mat H, FH, FHI;
	cv::Mat OUT_H;
	int size[] = { temp.rows, temp.cols };
	cv::Mat FFZP(2, size, CV_32FC2, cv::Scalar(0));
	cv::Mat FFZP2(2, size, CV_32FC2, cv::Scalar(0));
	sigma = CV_PI / (_cfgSig.lambda[0] * depth);
	sigmaf = (depth * _cfgSig.lambda[0]) / (4 * CV_PI);
	
	float row, col, color;
	row = temp.size[0];
	col = temp.size[1];
	color = temp.dims;

	int size1[] = { row };
	int size2[] = { col };
	cv::Mat r(1, size1, CV_32FC2, cv::Scalar(0));
	cv::Mat c(1, size2, CV_32FC2, cv::Scalar(0));

	c = r * 3;
	r = this->linspace(1, row, row);
	c = this->linspace(1, col, col);
	for (int i = 0; i < r.rows; i++)
	{
		r.at<float>(i) = (2 * CV_PI*(r.at<float>(i) - 1) / _cfgSig.width - CV_PI*(row - 1) / _cfgSig.width);
	}
			
	for (int i = 0; i < c.rows; i++)
	{
		c.at<float>(i) = (2 * CV_PI*(c.at<float>(i) - 1) / _cfgSig.height - CV_PI*(row - 1) / _cfgSig.height);
	}
	this->meshgrid(r, c, kx, ky);
	
	this->mul(kx, kx, kx);
	this->mul(ky, ky, ky);
	temp2.create(kx.rows, kx.cols, CV_32FC1);


	this->add(kx, ky, temp2);
	temp3.create(temp2.rows, temp2.cols, CV_32FC2);


	for (int i = 0; i < temp3.rows; i++)
	{
		for (int j = 0; j < temp3.cols; j++)
		{
			temp3.at<std::complex<float>>(i, j)._Val[0] = 0;
			temp3.at<std::complex<float>>(i, j)._Val[1] = temp2.at<float>(i, j);
		}
	}
	temp3 = temp3 * sigmaf;
	this->exp(temp3, FFZP);
	this->fftshift2d(FFZP, FFZP2);


	FH.create(temp.rows, temp.cols, CV_32FC2);
	this->fft2d(temp, FH);
	FHI.create(temp.rows, temp.cols, CV_32FC2);
	this->mul(FH, FFZP2, FHI);

	H.create(temp.rows, temp.cols, CV_32FC2);
	this->ifft2d(FHI, H);

	return H;	
}


void ophSig::ophFree(void) {

}