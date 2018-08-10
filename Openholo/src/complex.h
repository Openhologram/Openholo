#ifndef __complex_h
#define __complex_h

#include <iostream>
#include <cmath>
#include <complex>

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
			if (std::is_same<double, T>::value){
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

		const Complex<T>& operator()(T re, T im){
			_Val[_RE] = re;
			_Val[_IM] = im;

			return *this;
		}

		// arithmetic
		const Complex<T>& operator= (const Complex<T>& p) {
			_Val[_RE] = p._Val[_RE];
			_Val[_IM] = p._Val[_IM];

			return *this;
		}

		const Complex<T>& operator = (const T& p) {
			_Val[_RE] = p;

			return *this;
		}

		const Complex<T> operator+ (const Complex<T>& p) {
			Complex<T> n = *this + p;

			return n;
		}

		const Complex<T>& operator+= (const Complex<T>& p) {
			_Val[_RE] += p._Val[_RE];
			_Val[_IM] += p._Val[_IM];

			return *this;
		}

		const Complex<T> operator+ (const T p) {
			Complex<T> n(_Val[_RE] + p, _Val[_IM]);

			return n;
		}

		const Complex<T>& operator+= (const T p) {
			_Val[_RE] += p;

			return *this;
		}

		const Complex<T>& operator- (const Complex<T>& p) {
			Complex<T> n(_Val[_RE] - p._Val[_RE], _Val[_IM] - p._Val[_IM]);

			return n;
		}

		const Complex<T>& operator-= (const Complex<T>& p) {
			_Val[_RE] -= p._Val[_RE];
			_Val[_IM] -= p._Val[_IM];

			return *this;
		}

		const Complex<T> operator - (const T p) {
			Complex<T> n(_Val[_RE] - p, _Val[_IM]);

			return n;
		}

		const Complex<T>& operator -= (const T p) {
			_Val[_RE] -= p;

			return *this;
		}

		const Complex<T> operator* (const T k) {
			Complex<T> n(_Val[_RE] * k, _Val[_IM] * k);

			return *this;
		}

		const Complex<T>& operator*= (const T k) {
			_Val[_RE] *= k;
			_Val[_IM] *= k;

			return *this;
		}

		const Complex<T>& operator = (const std::complex<T> p) {
			_Val[_RE] = p._Val[_RE];
			_Val[_IM] = p._Val[_IM];

			return *this;
		}

		const Complex<T> operator* (const Complex<T>& p) {
			const T tRe = _Val[_RE];
			const T tIm = _Val[_IM];

			Complex<T> n(tRe * p._Val[_RE] - tIm * p._Val[_IM], tRe * p._Val[_IM] + tIm * p._Val[_RE]);

			return n;
		}

		const Complex<T>& operator*= (const Complex<T>& p) {
			const T tRe = _Val[_RE];
			const T tIm = _Val[_IM];

			_Val[_RE] = tRe * p._Val[_RE] - tIm * p._Val[_IM];
			_Val[_IM] = tRe * p._Val[_IM] + tIm * p._Val[_RE];

			return *this;
		}

		const Complex<T> operator / (const T& p) {
			Complex<T> n(_Val[_RE] / p, _Val[_IM] / p);

			return n;
		}

		const Complex<T>& operator/= (const T k) {
			_Val[_RE] /= k;
			_Val[_IM] /= k;

			return *this;
		}

		const Complex<T> operator / (const Complex<T>& p) {
			complex<T> a(_Val[_RE], _Val[_IM]);
			complex<T> b(p._Val[_RE], p._Val[_IM]);

			complex<T> c = a / b;

			Complex<T> n(c._Val[_RE], c._Val[_IM]);

			return n;
		}

		const Complex<T>& operator /= (const Complex<T>& p) {
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

		bool operator < (const Complex<T>& p){
			return (_Val[_RE] < p._Val[_RE]);
		}

		bool operator > (const Complex<T>& p) {
			return (_Val[_RE] > p._Val[_RE]);
		}

		operator unsigned char() {
			return unsigned char(_Val[_RE]);
		}

		operator int() {
			return int(_Val[_RE]);
		}

		friend const Complex<T> operator+ (const Complex<T>&p, const T q) {
			return Complex<T>(p._Val[_RE] + q, p._Val[_IM]);
		}

		friend const Complex<T> operator- (const Complex<T>&p, const T q) {
			return Complex<T>(p._Val[_RE] - q, p._Val[_IM]);
		}

		friend const Complex<T> operator* (const T k, const Complex<T>& p){
			return Complex<T>(p) *= k;
		}

		friend const Complex<T> operator* (const Complex<T>& p, const T k){
			return Complex<T>(p) *= k;
		}

		friend const Complex<T> operator* (const Complex<T>& p, const Complex<T>& q){
			return Complex<T>(p._Val[_RE]*q._Val[_RE] - p._Val[_IM]*q._Val[_IM], p._Val[_RE]*q._Val[_IM] + p._Val[_IM]*q._Val[_RE]);
		}

		friend const Complex<T> operator/ (const Complex<T>& p, const Complex<T>& q){
			return Complex<T>((1.0 / q.mag2())*(p*q.conj()));
		}

		friend const Complex<T> operator/ (const Complex<T>& p, const T& q) {
			return Complex<T>(p._Val[_RE] / q, p._Val[_IM]);
		}

		friend const Complex<T> operator/ (const T& p, const Complex<T>& q) {
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
