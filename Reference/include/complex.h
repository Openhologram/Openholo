
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
	*/
	class __declspec(dllexport) Complex
	{
	public:
		Complex() : a(0.0), b(0.0) {}
		Complex(double ta, double tb) : a(ta), b(tb) {}
		Complex(const Complex& p)
		{
			a = p.a;
			b = p.b;
		}

		double mag2() const { return a * a + b * b; }
		double mag()  const { return sqrt(a*a + b * b); }

		double arg() const
		{
			double r = mag();
			double theta = acos(a / r);

			if (sin(theta) - b / r < 10e-6)
				return theta;
			else
				return 2.0*PI - theta;
		}

		void euler(double& r, double& theta)
		{
			r = mag();
			theta = arg();
		}

		Complex conj() const { return Complex(a, -b); }

		// arithmetic
		const Complex& operator= (const Complex& p)
		{
			a = p.a;
			b = p.b;

			return *this;
		}

		const Complex& operator+= (const Complex& p)
		{
			a += p.a;
			b += p.b;

			return *this;
		}

		const Complex& operator-= (const Complex& p)
		{
			a -= p.a;
			b -= p.b;

			return *this;
		}

		const Complex& operator*= (const double k)
		{
			a *= k;
			b *= k;

			return *this;
		}

		const Complex& operator*= (const Complex& p)
		{
			const double ta = a;
			const double tb = b;

			a = ta * p.a - tb * p.b;
			b = ta * p.b + tb * p.a;

			return *this;
		}

		const Complex& operator/= (const double k)
		{
			a /= k;
			b /= k;

			return *this;
		}

		friend const Complex operator+ (const Complex& p, const Complex& q)
		{
			return Complex(p) += q;
		}

		friend const Complex operator- (const Complex& p, const Complex& q)
		{
			return Complex(p) -= q;
		}

		friend const Complex operator* (const double k, const Complex& p)
		{
			return Complex(p) *= k;
		}

		friend const Complex operator* (const Complex& p, const double k)
		{
			return Complex(p) *= k;
		}

		friend const Complex operator* (const Complex& p, const Complex& q)
		{
			return Complex(p.a*q.a - p.b*q.b, p.a*q.b + p.b*q.a);
		}

		friend const Complex operator/ (const Complex& p, const Complex& q)
		{
			return Complex((1.0 / q.mag2())*(p*q.conj()));
		}

		// stream
		friend std::ostream& operator << (std::ostream& os, const Complex& p)
		{
			os << "(" << p.a << ", " << p.b << ")";
			return os;
		}

	public:
		double a, b;
	};
}


#endif // !__complex_h_
