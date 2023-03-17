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
#include <complex> // c11 standard
#include "define.h"

#ifndef _RE
#define _RE 0
#endif
#ifndef _IM
#define _IM 1
#endif

#pragma once
#ifdef OPH_EXPORT
#ifdef _WIN32
#define OPH_DLL __declspec(dllexport)
#else
#define OPH_DLL __attribute__((visibility("default")))
#endif
#else
#define OPH_DLL
#endif

namespace oph {
	/**
	* @brief class for the complex number and its arithmetic.
	*		 type T == type cplx
	*		 type only float || double
	*		 T real() : real number
	*		 T imag() : imaginary number
	*/

#if _MSC_VER > 1900
	template<typename T = double>
	class OPH_DLL Complex : public std::complex<T>
	{
	public:
		using cplx = typename std::enable_if<std::is_same<double, T>::value || std::is_same<float, T>::value || std::is_same<long double, T>::value, T>::type;

	public:
		Complex() : std::complex<T>() {}
		Complex(T p) : std::complex<T>(p) { this->real(p); this->imag((T)0.0); }
		Complex(T tRe, T tIm) : std::complex<T>(tRe, tIm) {}
		Complex(const Complex<T>& p) {
			this->real(p.real());
			this->imag(p.imag());
		}
		T mag2() const { return this->real() * this->real() + this->imag() * this->imag(); }
		T mag()  const { return sqrt(mag2()); }

		T arg() const
		{
			T r = mag();
			T theta = acos(this->real() / r);

			if (sin(theta) - this->imag() / r < 10e-6)
				return theta;
			else
				return 2.0 * M_PI - theta;
		}

		void euler(T& r, T& theta)
		{
			r = mag();
			theta = arg();
		}

		T angle(void)
		{
			if (std::is_same<double, T>::value)
				return atan2(this->imag(), this->real());
			else if (std::is_same<float, T>::value)
				return atan2f((float)this->imag(), (float)this->real());
		}

		Complex<T>& exp() {
			Complex<T> p(this->real(), this->imag());
			if (std::is_same<double, T>::value) {
				this->real(std::exp(p.real()) * cos(p.imag()));
				this->imag(std::exp(p.real()) * sin(p.imag()));
			}
			else {
#ifdef _MSC_VER
				this->real(std::expf(p.real()) * cos(p.imag()));
				this->imag(std::expf(p.real()) * sin(p.imag()));
#else
				this->real(std::exp(p.real()) * cos(p.imag()));
				this->imag(std::exp(p.real()) * sin(p.imag()));
#endif
			}
			return *this;
		}

		Complex<T> conj() const { return Complex<T>(this->real(), -this->imag()); }

		Complex<T>& operator()(T re, T im) {
			this->real(re);
			this->imag(im);

			return *this;
		}

		// arithmetic
		Complex<T>& operator= (const Complex<T>& p) {
			this->real(p.real());
			this->imag(p.imag());

			return *this;
		}

		Complex<T>& operator = (const T& p) {
			this->real(p);
			this->imag(0.0);

			return *this;
		}

		Complex<T> operator+ (const Complex<T>& p) {
			Complex<T> n(this->real() + p.real(), this->imag() + p.imag());

			return n;
		}

		Complex<T>& operator+= (const Complex<T>& p) {
			this->real(this->real() + p.real());
			this->imag(this->imag() + p.imag());

			return *this;
		}

		Complex<T> operator+ (const T p) {
			Complex<T> n(this->real() + p, this->imag());

			return n;
		}

		Complex<T>& operator+= (const T p) {
			this->real(this->real() + p);

			return *this;
		}

		Complex<T> operator- (const Complex<T>& p) {
			Complex<T> n(this->real() - p.real(), this->imag() - p.imag());

			return n;
		}

		Complex<T>& operator-= (const Complex<T>& p) {
			this->real(this->real() - p.real());
			this->imag(this->imag() - p.imag());

			return *this;
		}

		Complex<T> operator - (const T p) {
			Complex<T> n(this->real() - p, this->imag());

			return n;
		}

		Complex<T>& operator -= (const T p) {
			this->real(this->real() - p);

			return *this;
		}

		Complex<T> operator* (const T k) {
			Complex<T> n(this->real() * k, this->imag() * k);

			return n;
		}

		Complex<T>& operator*= (const T k) {
			this->real(this->real() * k);
			this->imag(this->imag() * k);

			return *this;
		}

		Complex<T>& operator = (const std::complex<T>& p) {
			this->real(p.real());
			this->imag(p.imag());

			return *this;
		}

		Complex<T> operator* (const Complex<T>& p) {
			const T tRe = this->real();
			const T tIm = this->imag();

			Complex<T> n(tRe * p.real() - tIm * p.imag(), tRe * p.imag() + tIm * p.real());

			return n;
		}

		Complex<T>& operator*= (const Complex<T>& p) {
			const T tRe = this->real();
			const T tIm = this->imag();

			this->real(tRe * p.real() - tIm * p.imag());
			this->imag(tRe * p.imag() + tIm * p.real());

			return *this;
		}

		Complex<T> operator / (const T& p) {
			Complex<T> n(this->real() / p, this->imag() / p);

			return n;
		}

		Complex<T>& operator/= (const T k) {
			this->real(this->real() / k);
			this->imag(this->imag() / k);

			return *this;
		}

		Complex<T> operator / (const Complex<T>& p) {
			std::complex<T> a(this->real(), this->imag());
			std::complex<T> b(p.real(), p.imag());

			std::complex<T> c = a / b;

			Complex<T> n(c.real(), c.imag());

			return n;
		}

		Complex<T>& operator /= (const Complex<T>& p) {
			std::complex<T> a(this->real(), this->imag());
			std::complex<T> b(p.real(), p.imag());

			a /= b;
			this->real(a.real());
			this->imag(a.imag());

			return *this;
		}

		T& operator [](const int idx) {
			return reinterpret_cast<T*>(this)[idx];
		}

		bool operator < (const Complex<T>& p) {
			return (this->real() < p.real());
		}

		bool operator > (const Complex<T>& p) {
			return (this->real() > p.real());
		}

		operator unsigned char() {
			return uchar(this->real());
		}

		operator int() {
			return int(this->real());
		}

		friend Complex<T> operator+ (const Complex<T>&p, const T q) {
			return Complex<T>(p.real() + q, p.imag());
		}

		friend Complex<T> operator- (const Complex<T>&p, const T q) {
			return Complex<T>(p.real() - q, p.imag());
		}

		friend Complex<T> operator* (const T k, const Complex<T>& p) {
			return Complex<T>(p) *= k;
		}

		friend Complex<T> operator* (const Complex<T>& p, const T k) {
			return Complex<T>(p) *= k;
		}

		friend Complex<T> operator* (const Complex<T>& p, const Complex<T>& q) {
			return Complex<T>(p.real() * q.real() - p.imag() * q.imag(), p.real() * q.imag() + p.imag() * q.real());
		}

		friend Complex<T> operator/ (const Complex<T>& p, const Complex<T>& q) {
			return Complex<T>((1.0 / q.mag2())*(p*q.conj()));
		}

		friend Complex<T> operator/ (const Complex<T>& p, const T& q) {
			return Complex<T>(p.real() / q, p.imag() / q);
		}

		friend Complex<T> operator/ (const T& p, const Complex<T>& q) {
			return Complex<T>(p / q.real(), p / q.imag());
		}

		friend bool operator< (const Complex<T>& p, const Complex<T>& q) {
			return (p.real() < q.real());
		}

		friend bool operator> (const Complex<T>& p, const Complex<T>& q) {
			return (p.real() > q.real());
		}
	};

#else
template<typename T = double>
class OPH_DLL Complex : public std::complex<T>
{
public:
	using cplx = typename std::enable_if<std::is_same<double, T>::value || std::is_same<float, T>::value || std::is_same<long double, T>::value, T>::type;

public:
	Complex() : std::complex<T>() {}
	Complex(T p) : std::complex<T>(p) { this->_Val[_RE] = p; this->_Val[_IM] = (T)0.0; }
	Complex(T tRe, T tIm) : std::complex<T>(tRe, tIm) {}
	Complex(const Complex<T>& p) {
		this->_Val[_RE] = p._Val[_RE];
		this->_Val[_IM] = p._Val[_IM];
	}
	T mag2() const { return this->_Val[_RE] * this->_Val[_RE] + this->_Val[_IM] * this->_Val[_IM]; }
	T mag()  const { return sqrt(mag2()); }

	T arg() const
	{
		T r = mag();
		T theta = acos(this->_Val[_RE] / r);

		if (sin(theta) - this->_Val[_IM] / r < 10e-6)
			return theta;
		else
			return 2.0 * M_PI - theta;
	}

	void euler(T& r, T& theta)
	{
		r = mag();
		theta = arg();
	}

	T angle(void)
	{
		if (std::is_same<double, T>::value)
			return atan2(this->_Val[_IM], this->_Val[_RE]);
		else if (std::is_same<float, T>::value)
			return atan2f((float)this->_Val[_IM], (float)this->_Val[_RE]);
	}

	Complex<T>& exp() {
		Complex<T> p(this->_Val[_RE], this->_Val[_IM]);
		if (std::is_same<double, T>::value) {
			this->_Val[_RE] = std::exp(p._Val[_RE]) * cos(p._Val[_IM]);
			this->_Val[_IM] = std::exp(p._Val[_RE]) * sin(p._Val[_IM]);
		}
		else {
#ifdef _MSC_VER
			this->_Val[_RE] = std::expf(p._Val[_RE]) * cos(p._Val[_IM]);
			this->_Val[_IM] = std::expf(p._Val[_RE]) * sin(p._Val[_IM]);
#else
			this->_Val[_RE] = std::exp(p._Val[_RE]) * cos(p._Val[_IM]);
			this->_Val[_IM] = std::exp(p._Val[_RE]) * sin(p._Val[_IM]);
#endif
		}
		return *this;
	}

	Complex<T> conj() const { return Complex<T>(this->_Val[_RE], -this->_Val[_IM]); }

	Complex<T>& operator()(T re, T im) {
		this->_Val[_RE] = re;
		this->_Val[_IM] = im;

		return *this;
	}

	// arithmetic
	Complex<T>& operator= (const Complex<T>& p) {
		this->_Val[_RE] = p._Val[_RE];
		this->_Val[_IM] = p._Val[_IM];

		return *this;
	}

	Complex<T>& operator = (const T& p) {
		this->_Val[_RE] = p;
		this->_Val[_IM] = 0.0;

		return *this;
	}

	Complex<T> operator+ (const Complex<T>& p) {
		Complex<T> n(this->_Val[_RE] + p._Val[_RE], this->_Val[_IM] + p._Val[_IM]);

		return n;
	}

	Complex<T>& operator+= (const Complex<T>& p) {
		this->_Val[_RE] = this->_Val[_RE] + p._Val[_RE];
		this->_Val[_IM] = this->_Val[_IM] + p._Val[_IM];

		return *this;
	}

	Complex<T> operator+ (const T p) {
		Complex<T> n(this->_Val[_RE] + p, this->_Val[_IM]);

		return n;
	}

	Complex<T>& operator+= (const T p) {
		this->_Val[_RE] = this->_Val[_RE] + p;

		return *this;
	}

	Complex<T> operator- (const Complex<T>& p) {
		Complex<T> n(this->_Val[_RE] - p._Val[_RE], this->_Val[_IM] - p._Val[_IM]);

		return n;
	}

	Complex<T>& operator-= (const Complex<T>& p) {
		this->_Val[_RE] = this->_Val[_RE] - p._Val[_RE];
		this->_Val[_IM] = this->_Val[_IM] - p._Val[_IM];

		return *this;
	}

	Complex<T> operator - (const T p) {
		Complex<T> n(this->_Val[_RE] - p, this->_Val[_IM]);

		return n;
	}

	Complex<T>& operator -= (const T p) {
		this->_Val[_RE] = this->_Val[_RE] - p;

		return *this;
	}

	Complex<T> operator* (const T k) {
		Complex<T> n(this->_Val[_RE] * k, this->_Val[_IM] * k);

		return n;
	}

	Complex<T>& operator*= (const T k) {
		this->_Val[_RE] = this->_Val[_RE] * k;
		this->_Val[_IM] = this->_Val[_IM] * k;

		return *this;
	}

	Complex<T>& operator = (const std::complex<T>& p) {
		this->_Val[_RE] = p._Val[_RE];
		this->_Val[_IM] = p._Val[_IM];

		return *this;
	}

	Complex<T> operator* (const Complex<T>& p) {
		const T tRe = this->_Val[_RE];
		const T tIm = this->_Val[_IM];

		Complex<T> n(tRe * p._Val[_RE] - tIm * p._Val[_IM], tRe * p._Val[_IM] + tIm * p._Val[_RE]);

		return n;
	}

	Complex<T>& operator*= (const Complex<T>& p) {
		const T tRe = this->_Val[_RE];
		const T tIm = this->_Val[_IM];

		this->_Val[_RE] = tRe * p._Val[_RE] - tIm * p._Val[_IM];
		this->_Val[_IM] = tRe * p._Val[_IM] + tIm * p._Val[_RE];

		return *this;
	}

	Complex<T> operator / (const T& p) {
		Complex<T> n(this->_Val[_RE] / p, this->_Val[_IM] / p);

		return n;
	}

	Complex<T>& operator/= (const T k) {
		this->_Val[_RE] = this->_Val[_RE] / k;
		this->_Val[_IM] = this->_Val[_IM] / k;

		return *this;
	}

	Complex<T> operator / (const Complex<T>& p) {
		std::complex<T> a(this->_Val[_RE], this->_Val[_IM]);
		std::complex<T> b(p._Val[_RE], p._Val[_IM]);

		std::complex<T> c = a / b;

		Complex<T> n(c._Val[_RE], c._Val[_IM]);

		return n;
	}

	Complex<T>& operator /= (const Complex<T>& p) {
		std::complex<T> a(this->_Val[_RE], this->_Val[_IM]);
		std::complex<T> b(p._Val[_RE], p._Val[_IM]);

		a /= b;
		this->_Val[_RE] = a._Val[_RE];
		this->_Val[_IM] = a._Val[_IM];

		return *this;
	}

	T& operator [](const int idx) {
		return reinterpret_cast<T*>(this)[idx];
	}

	bool operator < (const Complex<T>& p) {
		return (this->_Val[_RE] < p._Val[_RE]);
	}

	bool operator > (const Complex<T>& p) {
		return (this->_Val[_RE] > p._Val[_RE]);
	}

	operator unsigned char() {
		return uchar(this->_Val[_RE]);
	}

	operator int() {
		return int(this->_Val[_RE]);
	}

	friend Complex<T> operator+ (const Complex<T>& p, const T q) {
		return Complex<T>(p._Val[_RE] + q, p._Val[_IM]);
	}

	friend Complex<T> operator- (const Complex<T>& p, const T q) {
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
		return Complex<T>((1.0 / q.mag2()) * (p * q.conj()));
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




#endif
}


#endif // !__complex_h_
