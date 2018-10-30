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

#ifndef __complex_h
#define __complex_h

#include <iostream>
#include <cmath>
#include <complex>

#ifndef _RE
#define _RE 0
#endif
#ifndef _IM
#define _IM 1
#endif

//#include "typedef.h"

namespace oph {
	/**
	* @brief class for the complex number and its arithmetic.
	*		 type T == type cplx
	*		 type only float || double
	*		 T _Val[_RE] : real number
	*		 T _Val[_IM] : imaginary number
	*/
	template<typename T = double>
	class __declspec(dllexport) Complex : public std::complex<T>
	{
	public:
		using cplx = typename std::enable_if<std::is_same<double, T>::value || std::is_same<float, T>::value || std::is_same<long double, T>::value, T>::type;

	public:
		Complex() : complex<T>() {}
		Complex(T p) : complex<T>(p) { _Val[_RE] = p; _Val[_IM] = 0.0; }
		Complex(T tRe, T tIm) : complex<T>(tRe, tIm) {}
		Complex(const Complex<T>& p) {
			_Val[_RE] = p._Val[_RE];
			_Val[_IM] = p._Val[_IM];
		}

		T mag2() const { return _Val[_RE] * _Val[_RE] + _Val[_IM] * _Val[_IM]; }
		T mag()  const { return sqrt(_Val[_RE] * _Val[_RE] + _Val[_IM] * _Val[_IM]); }

		T arg() const
		{
			T r = mag();
			T theta = acos(_Val[_RE] / r);

			if (sin(theta) - _Val[_IM] / r < 10e-6)
				return theta;
			else
				return 2.0*PI - theta;
		}

		void euler(T& r, T& theta)
		{
			r = mag();
			theta = arg();
		}

		T angle(void)
		{
			if (std::is_same<double, T>::value)
				return atan2(_Val[_IM], _Val[_RE]);
			else if (std::is_same<float, T>::value)
				return atan2f(_Val[_IM], _Val[_RE]);
		}

		Complex<T>& exp() {
			Complex<T> p(_Val[_RE], _Val[_IM]);
			if (std::is_same<double, T>::value) {
				_Val[_RE] = std::exp((float)p._Val[_RE]) * cos((float)p._Val[_IM]);
				_Val[_IM] = std::exp((float)p._Val[_RE]) * sin((float)p._Val[_IM]);
			}
			else {
				_Val[_RE] = std::expf(p._Val[_RE]) * cos(p._Val[_IM]);
				_Val[_IM] = std::expf(p._Val[_RE]) * sin(p._Val[_IM]);
			}
			return *this;
		}

		Complex<T> conj() const { return Complex<T>(_Val[_RE], -_Val[_IM]); }

		Complex<T>& operator()(T re, T im) {
			_Val[_RE] = re;
			_Val[_IM] = im;

			return *this;
		}

		// arithmetic
		Complex<T>& operator= (const Complex<T>& p) {
			_Val[_RE] = p._Val[_RE];
			_Val[_IM] = p._Val[_IM];

			return *this;
		}

		Complex<T>& operator = (const T& p) {
			_Val[_RE] = p;
			_Val[_IM] = 0.0;

			return *this;
		}

		Complex<T> operator+ (const Complex<T>& p) {
			Complex<T> n(_Val[_RE] + p._Val[_RE], _Val[_IM] + p._Val[_IM]);

			return n;
		}

		Complex<T>& operator+= (const Complex<T>& p) {
			_Val[_RE] += p._Val[_RE];
			_Val[_IM] += p._Val[_IM];

			return *this;
		}

		Complex<T> operator+ (const T p) {
			Complex<T> n(_Val[_RE] + p, _Val[_IM]);

			return n;
		}

		Complex<T>& operator+= (const T p) {
			_Val[_RE] += p;

			return *this;
		}

		Complex<T> operator- (const Complex<T>& p) {
			Complex<T> n(_Val[_RE] - p._Val[_RE], _Val[_IM] - p._Val[_IM]);

			return n;
		}

		Complex<T>& operator-= (const Complex<T>& p) {
			_Val[_RE] -= p._Val[_RE];
			_Val[_IM] -= p._Val[_IM];

			return *this;
		}

		Complex<T> operator - (const T p) {
			Complex<T> n(_Val[_RE] - p, _Val[_IM]);

			return n;
		}

		Complex<T>& operator -= (const T p) {
			_Val[_RE] -= p;

			return *this;
		}

		Complex<T> operator* (const T k) {
			Complex<T> n(_Val[_RE] * k, _Val[_IM] * k);

			return n;
		}

		Complex<T>& operator*= (const T k) {
			_Val[_RE] *= k;
			_Val[_IM] *= k;

			return *this;
		}

		Complex<T>& operator = (const std::complex<T>& p) {
			_Val[_RE] = p._Val[_RE];
			_Val[_IM] = p._Val[_IM];

			return *this;
		}

		Complex<T> operator* (const Complex<T>& p) {
			const T tRe = _Val[_RE];
			const T tIm = _Val[_IM];

			Complex<T> n(tRe * p._Val[_RE] - tIm * p._Val[_IM], tRe * p._Val[_IM] + tIm * p._Val[_RE]);

			return n;
		}

		Complex<T>& operator*= (const Complex<T>& p) {
			const T tRe = _Val[_RE];
			const T tIm = _Val[_IM];

			_Val[_RE] = tRe * p._Val[_RE] - tIm * p._Val[_IM];
			_Val[_IM] = tRe * p._Val[_IM] + tIm * p._Val[_RE];

			return *this;
		}

		Complex<T> operator / (const T& p) {
			Complex<T> n(_Val[_RE] / p, _Val[_IM] / p);

			return n;
		}

		Complex<T>& operator/= (const T k) {
			_Val[_RE] /= k;
			_Val[_IM] /= k;

			return *this;
		}

		Complex<T> operator / (const Complex<T>& p) {
			complex<T> a(_Val[_RE], _Val[_IM]);
			complex<T> b(p._Val[_RE], p._Val[_IM]);

			complex<T> c = a / b;

			Complex<T> n(c._Val[_RE], c._Val[_IM]);

			return n;
		}

		Complex<T>& operator /= (const Complex<T>& p) {
			complex<T> a(_Val[_RE], _Val[_IM]);
			complex<T> b(p._Val[_RE], p._Val[_IM]);

			a /= b;
			_Val[_RE] = a._Val[_RE];
			_Val[_IM] = a._Val[_IM];

			return *this;
		}

		T& operator [](const int idx) {
			return this->_Val[idx];
		}

		bool operator < (const Complex<T>& p) {
			return (_Val[_RE] < p._Val[_RE]);
		}

		bool operator > (const Complex<T>& p) {
			return (_Val[_RE] > p._Val[_RE]);
		}

		operator unsigned char() {
			return oph::uchar(_Val[_RE]);
		}

		operator int() {
			return int(_Val[_RE]);
		}

		friend Complex<T> operator+ (const Complex<T>&p, const T q) {
			return Complex<T>(p._Val[_RE] + q, p._Val[_IM]);
		}

		friend Complex<T> operator- (const Complex<T>&p, const T q) {
			return Complex<T>(p._Val[_RE] - q, p._Val[_IM]);
		}

		friend Complex<T> operator* (const T k, const Complex<T>& p) {
			return Complex<T>(p) *= k;
		}

		friend Complex<T> operator* (const Complex<T>& p, const T k) {
			return Complex<T>(p) *= k;
		}

		friend Complex<T> operator* (const Complex<T>& p, const Complex<T>& q) {
			return Complex<T>(p._Val[_RE] * q._Val[_RE] - p._Val[_IM] * q._Val[_IM], p._Val[_RE] * q._Val[_IM] + p._Val[_IM] * q._Val[_RE]);
		}

		friend Complex<T> operator/ (const Complex<T>& p, const Complex<T>& q) {
			return Complex<T>((1.0 / q.mag2())*(p*q.conj()));
		}

		friend Complex<T> operator/ (const Complex<T>& p, const T& q) {
			return Complex<T>(p._Val[_RE] / q, p._Val[_IM] / q);
		}

		friend Complex<T> operator/ (const T& p, const Complex<T>& q) {
			return Complex<T>(p / q._Val[_RE], p / q._Val[_IM]);
		}

		friend bool operator< (const Complex<T>& p, const Complex<T>& q) {
			return (p._Val[_RE] < q._Val[_RE]);
		}

		friend bool operator> (const Complex<T>& p, const Complex<T>& q) {
			return (p._Val[_RE] > q._Val[_RE]);
		}
	};
}


#endif // !__complex_h_
