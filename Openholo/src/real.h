#ifndef __real_h
#define __real_h

typedef double real;
typedef float  real_t;

#define REAL_T_IS_FLOAT 1

namespace oph {

#ifndef _MAXFLOAT
#define _MAXFLOAT	((float)3.40282347e+38)
#endif

#ifndef _MAXDOUBLE
#define _MAXDOUBLE	((double)1.7976931348623158e+308)
#endif

#define _MINFLOAT	((float)1.17549435e-38)
#define _MINDOUBLE	((double)2.2250738585072014e-308)

#ifndef M_PI
#define M_PI		 3.141592653589793238462643383279502884197169399375105820974944592308
#endif

#define MINREAL _MINDOUBLE;
#define MAXREAL _MAXDOUBLE;

	//template<class T = double>
	//struct real {
	//	using realnum = typename std::enable_if<std::is_same<double, T>::value || std::is_same<float, T>::value, T>::type;

	//	realnum t;

	//	real() { }
	//	real(realnum p) { t = p; }

	//	~real() { }

	//	bool operator == (const real<T>& p) { return t == p.t; }
	//	bool operator == (double p) { return t == p; }
	//	bool operator == (float p) { return t == p; }

	//	bool operator > (const real<T>& p) { return t > p.t; }
	//	bool operator > (double p) { return t > p; }
	//	bool operator > (float p) { return t > p; }

	//	bool operator >= (const real<T>& p) { return t >= p.t; }
	//	bool operator >= (double p) { return t > p; }
	//	bool operator >= (float p) { return t > p; }

	//	bool operator < (const real<T>& p) { return t < p.t; }
	//	bool operator < (double p) { return t < p; }
	//	bool operator < (float p) { return t < p; }

	//	bool operator <= (const real<T>& p) { return t <= p.t; }
	//	bool operator <= (double p) { return t <= p; }
	//	bool operator <= (float p) { return t <= p; }

	//	real<T>& operator = (const real<T>& b) { t = b.t; return *this; }
	//	real<T>& operator = (double b) { t = b; return *this; }
	//	real<T>& operator = (float b) { t = b; return *this; }

	//	realnum operator += (const real<T>& p) { return t += p.t; }
	//	double operator += (double p) { return t += p; }
	//	realnum operator += (float p) { return t += p; }

	//	realnum operator -= (const real<T>& p) { return t -= p.t; }
	//	double operator -= (double p) { return t -= p; }
	//	realnum operator -= (float p) { return t -= p; }

	//	realnum operator *= (const real<T>& p) { return t *= p.t; }
	//	double operator *= (double p) { return t *= p; }
	//	realnum operator *= (float p) { return t *= p; }

	//	realnum operator /= (const real<T>& p) { return t /= p.t; }
	//	double operator /= (double p) { return t /= p; }
	//	realnum operator /= (float p) { return t /= p; }

	//	realnum operator + (const real<T>& p) { return t + p.t; }
	//	realnum operator + (float p) { return t + p; }
	//	double operator + (double p) { return t + p; }

	//	realnum operator - (const real<T>& p) { return t - p.t; }
	//	double operator - (double p) { return t - p; }
	//	realnum operator - (float p) { return t - p; }

	//	realnum operator * (const real<T>& p) { return t * p.t; }
	//	double operator * (double p) { return t * p; }
	//	realnum operator * (float p) { return t * p; }

	//	realnum operator / (const real<T>& p) { return t / p.t; }
	//	double operator / (double p) { return t / p; }
	//	realnum operator / (float p) { return t / p; }

	//	realnum operator () () { return t; }

	//	friend bool operator < (double a, const real<T>&b) { return a < b.t; }
	//	friend bool operator <= (double a, const real<T>&b) { return a <= b.t; }
	//	//friend bool operator < (const real<T>&a, double b) { return a.t < b; }
	//	//friend bool operator <= (const real<T>&a, double b) { return a.t <= b; }

	//	friend bool operator > (double a, const real<T>&b) { return a > b.t; }
	//	friend bool operator >= (double a, const real<T>&b) { return a >= b.t; }
	//	//friend bool operator > (const real<T>&a, double b) { return a.t > b; }
	//	//friend bool operator >= (const real<T>&a, double b) { return a.t >= b; }

	//	friend bool operator == (double a, const real<T>&b) { return a == b.t; }

	//	friend double operator + (double a, const real<T>&b) { return a + b.t; }
	//	friend double operator - (double a, const real<T>&b) { return a - b.t; }
	//	friend double operator * (double a, const real<T>&b) { return a * b.t; }
	//	friend double operator / (double a, const real<T>&b) { return a / b.t; }
	//	friend T operator = (double a, const real<T>&b) { return a = b.t; }
	//	friend T operator = (float a, const real<T>&b) { return a = b.t; }

	//	friend std::ostream& operator << (std::ostream& os, const real<T>&p) { os << p.t; return os; }
	//	friend std::ostream& operator >> (std::ostream& os, const real<T>&p) { os >> p.t; return os; }
	//};


};

#endif // !__real_h
