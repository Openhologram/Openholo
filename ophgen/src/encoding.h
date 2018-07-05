#ifndef ENCODING_H
#define ENCODING_H

#define ENCODING_NUMERICAL_INTERFERENCE (1U << 0)
#define ENCODING_TWO_PHASE (1U << 1)
#define ENCODING_BURCKHARDT (1U << 2)
#define ENCODING_FREQ_SHIFT (1U << 3)


#define for_i(itr, oper) for(int i=0; i<itr; i++){ oper }

//void ophGen::calPhase(oph::Complex<oph::real>* holo, oph::real* encoded, const ivec2 holosize)
//{
//	int size = holosize.v[0] * holosize.v[1];
//	for_i(size,
//		oph::angle<oph::real>(*(holo + i), *(encoded + i));
//	);
//}
//void ophGen::calAmplitude(oph::Complex<oph::real>* holo, oph::real* encoded, const ivec2 holosize) {
//	int size = holosize.v[0] * holosize.v[1];
//	oph::absCplxArr<oph::real>(holo, encoded, size);
//}
//
//void ophGen::phase2amplitude(oph::real* encoded, const int size) {
//
//}
//
//void ophGen::numericalInterference(oph::Complex<oph::real>* holo, oph::real* encoded, const int size)
//{
//	oph::real* temp1 = new oph::real[size];
//	oph::absCplxArr<oph::real>(holo, temp1, size);
//
//	oph::real* ref = new oph::real;
//	*ref = oph::maxOfArr<oph::real>(temp1, size);
//
//	oph::Complex<oph::real>* temp2 = new oph::Complex<oph::real>[size];
//	temp2 = holo;
//	for_i(size,
//		temp2[i].re += *ref;
//	);
//
//	oph::absCplxArr<oph::real>(temp2, encoded, size);
//	for (int i = size - 10; i < size; i++)	// for debugging
//		cout << "result(" << i << "): " << *(encoded + i) << endl;
//
//	delete[] temp1, temp2;
//	delete ref;
//}
//
///* ม฿บน
//void ophGen::singleSideBand(oph::Complex<oph::real>* holo, oph::real* encoded, const ivec2 holosize, int passband)
//{
//int size = holosize.v[0] * holosize.v[1];
//
//oph::Complex<oph::real>* AS = new oph::Complex<oph::real>[size];
//fft2(holosize.v[0], holosize.v[1], holo, AS, FFTW_FORWARD);
//
//for (int i = size - 10; i < size; i++)	// for debugging
//cout << "AS(" << i << "): " << (AS + i)->re << " + " << (AS + i)->im << "i " << endl;
//
//switch (passband)
//{
//case left:
//cout << "left" << endl;	// for debugging
//for (int i = 0; i < holosize.v[1]; i++)
//{
//for (int j = holosize.v[0] / 2; j < holosize.v[0]; j++)
//{ AS[i*holosize.v[0] + j] = 0; }
//}
//case rig:
//for (int i = 0; i < holosize.v[1]; i++)
//{
//for (int j = 0; j < holosize.v[0] / 2; j++)
//{ AS[i*holosize.v[0] + j] = 0; }
//}
//case top:
//for (int i = size / 2; i < size; i++)
//{
//AS[i] = 0;
//}
//case btm:
//for (int i = 0; i < size / 2; i++)
//{
//AS[i] = 0;
//}
//}
//
//oph::Complex<oph::real>* filtered = new oph::Complex<oph::real>[size];
//fft2(holosize.v[0], holosize.v[1], AS, filtered, FFTW_BACKWARD);
//for (int i = size - 10; i < size; i++)	// for debugging
//cout << "filtered(" << i << "): " << (filtered + i)->re << " + " << (filtered + i)->im << "i " << endl;
//
//oph::real* oph::realPart = new oph::real[size];
//oph::oph::realPart<oph::real>(filtered, oph::realPart, size);
//for (int i = size - 10; i < size; i++)	// for debugging
//cout << "oph::real(" << i << "): " << *(oph::realPart + i) << endl;
//
//oph::real *minoph::real = new oph::real;
//*minoph::real = oph::minOfArr(oph::realPart, size);
//cout << "min: " << *minoph::real << endl;
//
//oph::real* oph::realPos = new oph::real[size];
//for_i(size,
//*(oph::realPos + i) = *(oph::realPart + i) - *minoph::real;
//);
//for (int i = size - 10; i < size; i++)	// for debugging
//cout << "oph::real-min(" << i << "): " << *(oph::realPos + i) << endl;
//
//oph::real *maxoph::real = new oph::real;
//*maxoph::real = oph::maxOfArr(oph::realPos, size);
//for (int i = size - 10; i < size; i++)	// for debugging
//cout << "max(" << i << "): " << *(maxoph::real + i) << endl;
//
//for_i(size,
//*(encoded + i) = *(oph::realPos + i) / *maxoph::real;
//);
//for (int i = size - 10; i < size; i++)	// for debugging
//cout << "(oph::real-min)/max(" << i << "): " << *(encoded + i) << endl;
//
//delete[] AS, filtered, oph::realPart, oph::realPos;
//delete maxoph::real, minoph::real;
//}
//*/
//
//void ophGen::twoPhaseEncoding(oph::Complex<oph::real>* holo, oph::real* encoded, const int size)
//{
//	Complex<oph::real>* normCplx = new Complex<oph::real>[size];
//	oph::normalize<oph::real>(holo, normCplx, size);
//
//	oph::real* amplitude = new oph::real[size];
//	calAmplitude(normCplx, amplitude, size);
//
//	oph::real* phase = new oph::real[size];
//	calPhase(normCplx, phase, size);
//
//	for_i(size, *(phase + i) += M_PI;);
//
//	oph::real* delPhase = new oph::real[size];
//	for_i(size, *(delPhase + i) = acos(*(amplitude + i)););
//
//	for_i(size,
//		*(encoded + i * 2) = *(phase + i) + *(delPhase + i);
//	*(encoded + i * 2 + 1) = *(phase + i) - *(delPhase + i);
//	);
//
//	delete[] normCplx, amplitude, phase, delPhase;
//}
//
//void ophGen::burckhardt(oph::Complex<oph::real>* holo, oph::real* encoded, const int size)
//{
//	Complex<oph::real>* norm = new Complex<oph::real>[size];
//	oph::normalize(holo, norm, size);
//	for (int i = 0; i < size; i++) {				// for debugging
//		cout << "norm(" << i << ": " << *(norm + i) << endl;
//	}
//	oph::real* phase = new oph::real[size];
//	calPhase(holo, phase, size);
//
//	for (int i = 0; i < size; i++) {				// for debugging
//		cout << "phase(" << i << ": " << *(phase + i) << endl;
//	}
//
//	oph::real* ampl = new oph::real[size];
//	calAmplitude(holo, ampl, size);
//
//	oph::real* A1 = new oph::real[size];
//	memsetArr<oph::real>(A1, 0, 0, size - 1);
//	oph::real* A2 = new oph::real[size];
//	memsetArr<oph::real>(A2, 0, 0, size - 1);
//	oph::real* A3 = new oph::real[size];
//	memsetArr<oph::real>(A3, 0, 0, size - 1);
//
//	for_i(size,
//		if (*(phase + i) >= 0 && *(phase + i) < (2 * M_PI / 3))
//		{
//			*(A1 + i) = *(ampl + i)*(cos(*(phase + i)) + sin(*(phase + i)) / sqrt(3));
//			*(A2 + i) = 2 * sin(*(phase + i)) / sqrt(3);
//			//cout << "A1,A2 : " << i << endl;
//		}
//		else if (*(phase + i) >= (2 * M_PI / 3) && *(phase + i) < (4 * M_PI / 3))
//		{
//			*(A2 + i) = *(ampl + i)*(cos(*(phase + i) - (2 * M_PI / 3)) + sin(*(phase + i) - (2 * M_PI / 3)) / sqrt(3));
//			*(A3 + i) = 2 * sin(*(phase + i) - (2 * M_PI / 3)) / sqrt(3);
//		}
//		else if (*(phase + i) >= (4 * M_PI / 3) && *(phase + i) < (2 * M_PI))
//		{
//			*(A3 + i) = *(ampl + i)*(cos(*(phase + i) - (4 * M_PI / 3)) + sin(*(phase + i) - (4 * M_PI / 3)) / sqrt(3));
//			*(A1 + i) = 2 * sin(*(phase + i) - (4 * M_PI / 3)) / sqrt(3);
//		}
//	);
//	for (int i = 0; i < size; i++) {				// for debugging
//		cout << "A1(" << i << ": " << *(A1 + i) << endl;
//		cout << "A2(" << i << ": " << *(A2 + i) << endl;
//		cout << "A3(" << i << ": " << *(A3 + i) << endl;
//	}
//	for_i(size / 3,
//		*(encoded + (3 * i)) = *(A1 + i);
//	*(encoded + (3 * i + 1)) = *(A2 + i);
//	*(encoded + (3 * i + 2)) = *(A3 + i);
//	);
//	for (int i = 0; i < size; i++) {				// for debugging
//		cout << "encoded(" << i << ": " << *(encoded + i) << endl;
//	}
//}
//
//void ophGen::freqShift(oph::Complex<oph::real>* holo, Complex<oph::real>* encoded, const ivec2 holosize, int shift_x, int shift_y)
//{
//	int size = holosize.v[0] * holosize.v[1];
//
//	oph::Complex<oph::real>* AS = new oph::Complex<oph::real>[size];
//	fft2(holosize.v[0], holosize.v[1], holo, AS, FFTW_FORWARD);
//	oph::Complex<oph::real>* shifted = new oph::Complex<oph::real>[size];
//	circshift<Complex<oph::real>>(AS, shifted, shift_x, shift_y, holosize.v[0], holosize.v[1]);
//	fft2(holosize.v[0], holosize.v[1], shifted, encoded, FFTW_BACKWARD);
//}


#endif