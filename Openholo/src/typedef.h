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
	typedef unsigned short ushort;
	typedef unsigned char uchar;
	typedef unsigned long ulong;
	typedef long long longlong;
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