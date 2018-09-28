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

#include "epsilon.h"
#include <math.h>
#include "sys.h"
#include "define.h"

namespace oph {

Real epsilon = 1.0e-8;
Real user_epsilon = 1.0e-8;
Real intersection_epsilon = 1e-6;
Real sqrt_epsilon =  1.490116119385000000e-8;
Real unset_value = -1.23432101234321e+308;
Real zero_tolerance = 1.0e-12;
Real zero_epsilon = 1.0e-12;
Real angle_tolerance = M_PI/180.0;
Real save_zero_epsilon = 1.0e-12;


/*|--------------------------------------------------------------------------*/
/*| Set user epsilon : Throughout the running program we could use the same  */
/*| user epsilon defined here. Default user_epsilon is always 1e-8.          */
/*|--------------------------------------------------------------------------*/
void set_u_epsilon(Real a)
{
    user_epsilon = a;
}

void reset_u_epsilon()
{
    user_epsilon = epsilon;
}
void set_zero_epsilon(Real a)
{
	save_zero_epsilon = zero_epsilon;
	zero_epsilon = a;
}

void reset_zero_epsilon()
{
	zero_epsilon = save_zero_epsilon;
}

/*|--------------------------------------------------------------------------*/
/*| Approximated version of checking equality : using epsilon                */
/*|--------------------------------------------------------------------------*/
int apx_equal(Real x, Real y)
{
    int c;
    Real a;

    a = Real(fabsf(((float)x) - ((float)y)));

    if (a < user_epsilon) c = 1;
    else c = 0;

    return c;
}

/*|--------------------------------------------------------------------------*/
/*| Approximated version of checking equality : using epsilon                */
/*|--------------------------------------------------------------------------*/
int apx_equal(Real x, Real y, Real eps)
{
    int c;
    Real a;

    a = Real(fabsf(((float)x) - ((float)y)));

    if (a < eps) c = 1;
    else c = 0;

    return c;
}
}; // namespace graphics