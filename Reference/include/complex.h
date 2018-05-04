
//-------------------------------------------------------------------------
// Complex Numbers
// Revision :   $Rev$
// Last changed : $Date$
//
// Author       : Seungtaik Oh
// Last Update  : 19 JUL 2011
//-------------------------------------------------------------------------
#ifndef __complex_h_
#define __complex_h_

#include <iostream>
#include <cmath>

const double PI = 3.141592653589793238462643383279502884197169399375105820974944592308;
const double TWO_PI = 2.0*PI;


namespace oph {
	/**
	* @brief class for the complex number and its arithmetic.
	*		 typename T equal typename cplx
	*		 type only float || double
	*		 cplx re : real number
	*		 cplx im : imaginary number
	*/
	template<typename T = double>
	class __declspec(dllexport) Complex
	{
	public:
		using cplx = typename std::enable_if<std::is_same<double, T>::value || std::is_same<float, T>::value, T>::type;

	public:
		cplx re, im;

		Complex() : re(0), im(0) {}
		Complex(cplx tRe, cplx tIm) : re(tRe), im(tIm) {}
		Complex(const Complex<T>& p)
		{
			re = p.re;
			im = p.im;
		}

		cplx mag2() const { return re * re + im * im; }
		cplx mag()  const { return sqrt(re * re + im * im); }

		cplx arg() const
		{
			cplx r = mag();
			cplx theta = acos(re / r);

			if (sin(theta) - im / r < 10e-6)
				return theta;
			else
				return 2.0*PI - theta;
		}

		void euler(cplx& r, cplx& theta)
		{
			r = mag();
			theta = arg();
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

		const Complex<T>& operator/= (const cplx k){
			re /= k;
			im /= k;

			return *this;
		}

		friend const Complex<T> operator+ (const Complex<T>& p, const Complex<T>& q){
			return Complex<T>(p) += q;
		}

		friend const Complex<T> operator- (const Complex<T>& p, const Complex<T>& q){
			return Complex<T>(p) -= q;
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

		// stream
		friend std::ostream& operator << (std::ostream& os, const Complex<T>& p){
			os << "(" << p.re << ", " << p.im << ")";
			return os;
		}
	};
}


#endif // !__complex_h_
