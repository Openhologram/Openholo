#ifndef __complex_h_
#define __complex_h_

#include <iostream>
#include <cmath>

#include "typedef.h"

namespace oph {
	/**
	* @brief class for the complex number and its arithmetic.
	*		 type T == type cplx
	*		 type only float || double
	*		 T re : real number
	*		 T im : imaginary number
	*/
	template<typename T = double>
	class __declspec(dllexport) Complex
	{
	public:
		using cplx = typename std::enable_if<std::is_same<double, T>::value || std::is_same<float, T>::value, T>::type;

	public:
		T re, im;

		Complex() : re(0), im(0) {}
		Complex(T p) : re(p), im(p) {}
		Complex(T tRe, T tIm) : re(tRe), im(tIm) {}
		Complex(const Complex<T>& p)
		{
			re = p.re;
			im = p.im;
		}

		T mag2() const { return re * re + im * im; }
		T mag()  const { return sqrt(re * re + im * im); }

		T arg() const
		{
			T r = mag();
			T theta = acos(re / r);

			if (sin(theta) - im / r < 10e-6)
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
		{matrix identity
			if (std::is_same<double, T>::value)
				return atan2(im, re);
			else if (std::is_same<float, T>::value)
				return atan2f(im, re);
		}

		Complex<T>& exp() {
			if (std::is_same<double, T>::value){
				re = std::exp(re) * cos(im);
				im = std::exp(re) * sin(im);
			}
			else {
				re = std::expf(re) * cos(im);
				im = std::expf(re) * sin(im);
			}
			return *this;
		}

		Complex<T> conj() const { return Complex<T>(re, -im); }

		const Complex<T>& operator()(T re, T im){
			re = re;
			im = im;

			return *this;
		}

		// arithmetic
		const Complex<T>& operator= (const Complex<T>& p){
			re = p.re;
			im = p.im;

			return *this;
		}

		const Complex<T>& operator = (const T& p) {
			re = p;
			im = p;

			return *this;
		}

		const Complex<T>& operator+= (const Complex<T>& p){
			re += p.re;
			im += p.im;

			return *this;
		}

		const Complex<T>& operator-= (const Complex<T>& p){
			re -= p.re;
			im -= p.im;

			return *this;
		}

		const Complex<T>& operator*= (const cplx k){
			re *= k;
			im *= k;

			return *this;
		}

		const Complex<T>& operator*= (const Complex<T>& p){
			const cplx tRe = re;
			const cplx tIm = im;

			re = tRe * p.re - tIm * p.im;
			im = tRe * p.im + tIm * p.re;

			return *this;
		}

		const Complex<T>& operator/= (const cplx k) {
			re /= k;
			im /= k;

			return *this;
		}

		const Complex<T>& operator /= (const Complex<T>& p) {
			re /= p.re;
			im /= p.im;

			return *this;
		}

		bool operator < (const Complex<T>& p){
			return (re < p.re);
		}

		bool operator > (const Complex<T>& p) {
			return (re > p.re);
		}

		operator unsigned char() {
			return unsigned char(re);
		}

		operator int() {
			return int(re);
		}

		friend const Complex<T> operator+ (const Complex<T>& p, const Complex<T>& q){
			return Complex<T>(p) += q;
		}

		friend const Complex<T> operator+ (const Complex<T>&p, const double q) {
			return Complex<T>(p.re + q, p.im);
		}

		friend const Complex<T> operator- (const Complex<T>& p, const Complex<T>& q){
			return Complex<T>(p) -= q;
		}

		friend const Complex<T> operator- (const Complex<T>&p, const double q) {
			return Complex<T>(p.re - q, p.im);
		}

		friend const Complex<T> operator* (const cplx k, const Complex<T>& p){
			return Complex<T>(p) *= k;
		}

		friend const Complex<T> operator* (const Complex<T>& p, const double k){
			return Complex<T>(p) *= k;
		}

		friend const Complex<T> operator* (const Complex<T>& p, const Complex<T>& q){
			return Complex<T>(p.re*q.re - p.im*q.im, p.re*q.im + p.im*q.re);
		}

		friend const Complex<T> operator/ (const Complex<T>& p, const Complex<T>& q){
			return Complex<T>((1.0 / q.mag2())*(p*q.conj()));
		}

		friend bool operator< (const Complex<T>& p, const Complex<T>& q) {
			return (p.re < q.re);
		}

		friend bool operator> (const Complex<T>& p, const Complex<T>& q) {
			return (p.re > q.re);
		}

		// stream
		friend std::ostream& operator << (std::ostream& os, const Complex<T>& p){
			os << "(" << p.re << ", " << p.im << ")";
			return os;
		}
	};
}


#endif // !__complex_h_
