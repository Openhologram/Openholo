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

#ifndef __define_h
#define __define_h

namespace oph
{
#ifndef M_PI
#define M_PI	3.141592653589793238462643383279502884197169399375105820974944592308
#endif

#ifndef M_PI_F
#define M_PI_F	3.14159265358979323846f
#endif

//Convert Angle double
#define RADIAN(theta)	(theta*M_PI)/180.0
#define DEGREE(theta)	(theta*180.0)/M_PI
//-				float
#define RADIAN_F(theta) (theta*M_PI_F)/180.f
#define DEGREE_F(theta) (theta*180.f)/M_PI_F

#define OPH_FORWARD (-1)
#define OPH_BACKWARD (1)

#define OPH_MEASURE (0U)
#define OPH_DESTROY_INPUT (1U << 0)
#define OPH_UNALIGNED (1U << 1)
#define OPH_CONSERVE_MEMORY (1U << 2)
#define OPH_EXHAUSTIVE (1U << 3)
#define OPH_PRESERVE_INPUT (1U << 4)
#define OPH_PATIENT (1U << 5)
#define OPH_ESTIMATE (1U << 6)
#define OPH_WISDOM_ONLY (1U << 21)

#ifndef _X
#define _X 0
#endif

#ifndef _Y
#define _Y 1
#endif

#ifndef _Z
#define _Z 2
#endif

#ifndef _W
#define _W 3
#endif

#ifndef _R
#define _R 0
#endif

#ifndef _G
#define _G 1
#endif

#ifndef _B
#define _B 2
#endif

#ifndef _COL
#define _COL 0
#endif

#ifndef _ROW
#define _ROW 1
#endif

#ifndef _MAT
#define _MAT 2
#endif

#ifndef MAX_FLOAT
#define MAX_FLOAT	((float)3.40282347e+38)
#endif

#ifndef MAX_DOUBLE
#define MAX_DOUBLE	((double)1.7976931348623158e+308)
#endif

#ifndef MIN_FLOAT
#define MIN_FLOAT	((float)1.17549435e-38)
#endif

#ifndef MIN_DOUBLE
#define MIN_DOUBLE	((double)2.2250738585072014e-308)
#endif

#define MIN_REAL MIN_DOUBLE;
#define MAX_REAL MAX_DOUBLE;

//Mode Flag
#define MODE_CPU		0
#define MODE_GPU		1
#define MODE_DOUBLE		0 // default
#define MODE_FLOAT		2
#define MODE_FASTMATH	4

#define WIDTHBYTES(bits) (((bits)+31)/32*4)

#define OPH_PLANES 1
#define OPH_COMPRESSION 0
#define X_PIXEL_PER_METER 0x130B //2835 , 72 DPI X (4875)
#define Y_PIXEL_PER_METER 0x130B //2835 , 72 DPI X (4875)
}

#endif // !__define_h