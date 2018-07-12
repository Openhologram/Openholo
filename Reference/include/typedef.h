#ifndef __typedef_h
#define __typedef_h

#define REAL_IS_DOUBLE true

#if REAL_IS_DOUBLE & true
typedef double Real;
typedef float  Real_t;
#else
typedef float Real;
typedef double Real_t;
#endif

namespace oph
{
	typedef unsigned int uint;
	typedef unsigned char uchar;
	typedef unsigned long ulong;
	typedef unsigned long long ulonglong;


	//typedef std::array<int, 2> int2;
	//typedef std::array<int, 3> int3;
	//typedef std::array<int, 4> int4;

	//typedef std::array<uint, 2> uint2;
	//typedef std::array<uint, 3> uint3;
	//typedef std::array<uint, 4> uint4;

	//typedef std::array<oph::real, 2> real2;
	//typedef std::array<oph::real, 3> real3;
	//typedef std::array<oph::real, 4> real4;

	//typedef std::array<oph::real_t, 2> real_t2;
	//typedef std::array<oph::real_t, 3> real_t3;
	//typedef std::array<oph::real_t, 4> real_t4;

	//typedef std::array<std::complex<real>, 2> complex2;
	//typedef std::array<std::complex<real>, 3> complex3;
	//typedef std::array<std::complex<real>, 4> complex4;

	//typedef std::array<std::complex<real_t>, 2> complex_t2;
	//typedef std::array<std::complex<real_t>, 3> complex_t3;
	//typedef std::array<std::complex<real_t>, 4> complex_t4;
}

#endif // !__typedef_h