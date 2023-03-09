#ifndef __real_h
#define __real_h



namespace oph {


	//template<class T = double>
	//struct real_number 
	//{
	//	//using realnum = typename std::enable_if<std::is_same<double, T>::value || std::is_same<float, T>::value, T>::type;

	//	T t;

	//	real_number() { }
	//	real_number(T p) { t = p; }

	//	~real_number() { }

	//	inline bool operator == (const real_number<T>& p) { return t == p.t; }
	//	inline bool operator == (double p) { return t == p; }
	//	inline bool operator == (float p) { return t == p; }
	//	
	//	inline bool operator > (const real_number<T>& p) { return t > p.t; }
	//	inline bool operator > (double p) { return t > p; }
	//	inline bool operator > (float p) { return t > p; }
	//	
	//	inline bool operator >= (const real_number<T>& p) { return t >= p.t; }
	//	inline bool operator >= (double p) { return t > p; }
	//	inline bool operator >= (float p) { return t > p; }
	//	
	//	inline bool operator < (const real_number<T>& p) { return t < p.t; }
	//	inline bool operator < (double p) { return t < p; }
	//	inline bool operator < (float p) { return t < p; }
	//	
	//	inline bool operator <= (const real_number<T>& p) { return t <= p.t; }
	//	inline bool operator <= (double p) { return t <= p; }
	//	inline bool operator <= (float p) { return t <= p; }
	//	
	//	inline real_number<T>& operator = (const real_number<T>& b) { t = b.t; return *this; }
	//	inline real_number<T>& operator = (double b) { t = b; return *this; }
	//	inline real_number<T>& operator = (float b) { t = b; return *this; }
	//	
	//	inline T operator += (const real_number<T>& p) { return t += p.t; }
	//	inline double operator += (double p) { return t += p; }
	//	inline T operator += (float p) { return t += p; }
	//	
	//	inline T operator -= (const real_number<T>& p) { return t -= p.t; }
	//	inline double operator -= (double p) { return t -= p; }
	//	inline T operator -= (float p) { return t -= p; }
	//	
	//	inline T operator *= (const real_number<T>& p) { return t *= p.t; }
	//	inline double operator *= (double p) { return t *= p; }
	//	inline T operator *= (float p) { return t *= p; }
	//	
	//	inline T operator /= (const real_number<T>& p) { return t /= p.t; }
	//	inline double operator /= (double p) { return t /= p; }
	//	inline T operator /= (float p) { return t /= p; }
	//	
	//	inline T operator + (const real_number<T>& p) { return t + p.t; }
	//	inline T operator + (float p) { return t + p; }
	//	inline double operator + (double p) { return t + p; }
	//	
	//	inline T operator - (const real_number<T>& p) { return t - p.t; }
	//	inline double operator - (double p) { return t - p; }
	//	inline T operator - (float p) { return t - p; }
	//
	//	inline T operator * (const real_number<T>& p) { return t * p.t; }
	//	inline double operator * (double p) { return t * p; }
	//	inline T operator * (float p) { return t * p; }

	//	inline T operator / (const real_number<T>& p) { return t / p.t; }
	//	inline double operator / (double p) { return t / p; }
	//	inline T operator / (float p) { return t / p; }

	//	inline T operator () () { return t; }

	//	inline friend bool operator < (double p, const real_number<T>&q) { return p < q.t; }
	//	inline friend bool operator <= (double p, const real_number<T>&q) { return p <= q.t; }
	//	inline friend bool operator > (double p, const real_number<T>&q) { return p > q.t; }
	//	inline friend bool operator >= (double p, const real_number<T>&q) { return p >= q.t; }
	//	
	//	inline friend bool operator < (float p, const real_number<T>&q) { return p < q.t; }
	//	inline friend bool operator <= (float p, const real_number<T>&q) { return p <= q.t; }
	//	inline friend bool operator > (float p, const real_number<T>&q) { return p > q.t; }
	//	inline friend bool operator >= (float p, const real_number<T>&q) { return p >= q.t; }
	//	
	//	inline friend bool operator == (double p, const real_number<T>&q) { return p == q.t; }
	//	
	//	inline friend double operator + (double p, const real_number<T>&q) { return p + q.t; }
	//	inline friend double operator - (double p, const real_number<T>&q) { return p - q.t; }
	//	inline friend double operator * (double p, const real_number<T>&q) { return p * q.t; }
	//	inline friend double operator / (double p, const real_number<T>&q) { return p / q.t; }
	//	
	//	inline friend double operator = (double p, const real_number<T>&q) { p = q.t; return p; }
	//	inline friend float operator = (float p, const real_number<T>&q) { p = q.t; return p; }
	//	
	//	inline friend std::ostream& operator << (std::ostream& os, const real_number<T>&p) { os << p.t; return os; }
	//	inline friend std::ostream& operator >> (std::ostream& os, const real_number<T>&p) { os >> p.t; return os; }
	//};

	//typedef real_number<double> real;
	//typedef real_number<float> real_t;

};

#endif // !__real_h
