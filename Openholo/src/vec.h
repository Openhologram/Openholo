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

#ifndef __vec_h
#define __vec_h
// Description:
//  Mathematical tools to handle n-dimensional vectors
//
// Author:
//   Myung-Joon Kim
//   Dae-Hyun Kim


#include "ivec.h"
#include "epsilon.h"
#include <math.h>
#include <stdio.h>

namespace oph {

	/**
	* @brief structure for 2-dimensional Real type vector and its arithmetic.
	*/
	struct __declspec(dllexport) vec2 {
		Real v[2];
		static const int n;

		inline vec2() { }
		inline vec2(Real a) { v[1 - 1] = a;  v[2 - 1] = a; }
		inline vec2(Real v_1, Real v_2) { v[1 - 1] = v_1;  v[2 - 1] = v_2; }
		inline vec2(const ivec2& a) { v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1]; }
		inline vec2(const vec2& a) { v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1]; }


		inline vec2& operator=(const vec2& a) { v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1]; return *this; }
		inline Real& operator[] (int i) { return v[i]; }
		inline const Real&  operator[] (int i) const { return v[i]; }
		inline Real& operator() (int i) { return v[i % 2]; }
		inline const Real&  operator() (int i) const { return v[i % 2]; }

		bool unit();
		Real length() const;

		inline bool is_zero() const { return (v[0] == 0.0 && v[1] == 0.0); }
		inline bool is_tiny(Real tiny_tol = epsilon) const {
			return (fabs(v[0]) <= tiny_tol && fabs(v[1]) <= tiny_tol);
		}

		//
		// returns  1: this and other vectors are parallel
		//         -1: this and other vectors are anti-parallel
		//          0: this and other vectors are not parallel
		//             or at least one of the vectors is zero
		int is_parallel(
			const vec2&,                 // other vector     
			Real = angle_tolerance // optional angle tolerance (radians)
		) const;

		// returns true:  this and other vectors are perpendicular
		//         false: this and other vectors are not perpendicular
		//                or at least one of the vectors is zero
		bool is_perpendicular(
			const vec2&,           // other vector     
			Real = angle_tolerance // optional angle tolerance (radians)
		) const;

		//
		// set this vector to be perpendicular to another vector
		bool perpendicular( // Result is not unitized. 
							 // returns false if input vector is zero
			const vec2&
		);

		//
		// set this vector to be perpendicular to a line defined by 2 points
		bool perpendicular(
			const vec2&,
			const vec2&
		);
	};





	//| binary op : componentwise


	inline vec2 operator + (const vec2& a, const vec2& b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a[i] + b[i]; }
		return c;
	}

	inline vec2 operator + (Real a, const vec2& b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a + b[i]; }
		return c;
	}

	inline vec2 operator + (const vec2& a, Real b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a[i] + b; }
		return c;
	}



	inline vec2 operator - (const vec2& a, const vec2& b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a[i] - b[i]; }
		return c;
	}

	inline vec2 operator - (Real a, const vec2& b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a - b[i]; }
		return c;
	}

	inline vec2 operator - (const vec2& a, Real b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a[i] - b; }
		return c;
	}



	inline vec2 operator * (const vec2& a, const vec2& b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a[i] * b[i]; }
		return c;
	}

	inline vec2 operator * (Real a, const vec2& b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a * b[i]; }
		return c;
	}

	inline vec2 operator * (const vec2& a, Real b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a[i] * b; }
		return c;
	}



	inline vec2 operator / (const vec2& a, const vec2& b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a[i] / b[i]; }
		return c;
	}

	inline vec2 operator / (Real a, const vec2& b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a / b[i]; }
		return c;
	}

	inline vec2 operator / (const vec2& a, Real b)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = a[i] / b; }
		return c;
	}



	//| cumulative op : componentwise


	inline vec2 operator += (vec2& a, const vec2& b)
	{
		return a = (a + b);
	}

	inline vec2 operator += (vec2& a, Real b)
	{
		return a = (a + b);
	}



	inline vec2 operator -= (vec2& a, const vec2& b)
	{
		return a = (a - b);
	}

	inline vec2 operator -= (vec2& a, Real b)
	{
		return a = (a - b);
	}



	inline vec2 operator *= (vec2& a, const vec2& b)
	{
		return a = (a * b);
	}

	inline vec2 operator *= (vec2& a, Real b)
	{
		return a = (a * b);
	}



	inline vec2 operator /= (vec2& a, const vec2& b)
	{
		return a = (a / b);
	}

	inline vec2 operator /= (vec2& a, Real b)
	{
		return a = (a / b);
	}



	//| logical op : componentwise


	inline int operator == (const vec2& a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] == b[i]; }
		return c;
	}

	inline int operator == (Real a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a == b[i]; }
		return c;
	}

	inline int operator == (const vec2& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] == b; }
		return c;
	}



	inline int operator < (const vec2& a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] < b[i]; }
		return c;
	}

	inline int operator < (Real a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a < b[i]; }
		return c;
	}

	inline int operator < (const vec2& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] < b; }
		return c;
	}



	inline int operator <= (const vec2& a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] <= b[i]; }
		return c;
	}

	inline int operator <= (Real a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a <= b[i]; }
		return c;
	}

	inline int operator <= (const vec2& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] <= b; }
		return c;
	}



	inline int operator > (const vec2& a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] > b[i]; }
		return c;
	}

	inline int operator > (Real a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a > b[i]; }
		return c;
	}

	inline int operator > (const vec2& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] > b; }
		return c;
	}



	inline int operator >= (const vec2& a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] >= b[i]; }
		return c;
	}

	inline int operator >= (Real a, const vec2& b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a >= b[i]; }
		return c;
	}

	inline int operator >= (const vec2& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 2; ++i) { c = c && a[i] >= b; }
		return c;
	}



	//| unary op : componentwise
	inline vec2 operator - (const vec2& a)
	{
		vec2 c;
		for (int i = 0; i < 2; ++i) { c[i] = -a[i]; }
		return c;
	}

	//| R^n -> R
	inline Real sum(const vec2& a)
	{
		Real s = 0;

		s += a[1 - 1];
		s += a[2 - 1];

		return s;
	}

	inline Real inner(const vec2& a, const vec2& b)
	{
		vec2 tmp = a * b;
		return sum(tmp);
	}

	inline Real norm(const vec2& a)
	{
		return sqrt(inner(a, a));
	}

	inline Real squaredNorm(const vec2& a) {
		return inner(a, a);
	}

	inline vec2 unit(const vec2& a)
	{
		Real n = norm(a);
		if (n < epsilon)
			return 0;
		else
			return a / n;
	}

	inline Real angle(const vec2& a, const vec2& b)
	{
		Real ang = inner(unit(a), unit(b));
		if (ang > 1 - epsilon)
			return 0;
		else if (ang < -1 + epsilon)
			return M_PI;
		else
			return acos(ang);
	}

	inline vec2 proj(const vec2& axis, const vec2& a)
	{
		vec2 u = unit(axis);
		return inner(a, u) * u;
	}

	inline vec2 absolute(const vec2& val)
	{
		return vec2(fabs(val[0]), fabs(val[1]));
	}

	void store(FILE* fp, const vec2& v);
	//int scan(FILE* fp, const vec2& v);

	int apx_equal(const vec2& a, const vec2& b);
	int apx_equal(const vec2& a, const vec2& b, Real eps);

	/**
	* @brief structure for 3-dimensional Real type vector and its arithmetic.
	*/
	struct __declspec(dllexport) vec3 {
		Real v[3];
		static const int n;

		inline vec3() { }
		inline vec3(Real a) { v[1 - 1] = a;  v[2 - 1] = a;  v[3 - 1] = a; }
		inline vec3(Real v_1, Real v_2, Real v_3) { v[1 - 1] = v_1;  v[2 - 1] = v_2;  v[3 - 1] = v_3; }
		inline vec3(const ivec3& a) { v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1]; }
		inline vec3(const vec3& a) { v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1]; }

		inline vec3& operator=(const vec3& a) { v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1]; return *this; }
		inline Real& operator[] (int i) { return v[i]; }
		inline const Real&  operator[] (int i) const { return v[i]; }
		inline Real& operator() (int i) { return v[i % 3]; }
		inline const Real&  operator() (int i) const { return v[i % 3]; }

		inline bool is_zero() const { return (v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0); }
		inline bool is_tiny(Real tiny_tol = epsilon) const { return (fabs(v[0]) <= tiny_tol && fabs(v[1]) <= tiny_tol && fabs(v[2]) <= tiny_tol); }

		bool unit();
		Real length() const;


		//
		// returns  1: this and other vectors are parallel
		//         -1: this and other vectors are anti-parallel
		//          0: this and other vectors are not parallel
		//             or at least one of the vectors is zero
		int is_parallel(
			const vec3&,                 // other vector     
			Real = angle_tolerance // optional angle tolerance (radians)
		) const;

		//
		// returns true:  this and other vectors are perpendicular
		//         false: this and other vectors are not perpendicular
		//                or at least one of the vectors is zero
		bool is_perpendicular(
			const vec3&,                 // other vector     
			Real = angle_tolerance // optional angle tolerance (radians)
		) const;

		//
		// set this vector to be perpendicular to another vector
		bool perpendicular( // Result is not unitized. 
							// returns false if input vector is zero
			const vec3&
		);

		//
		// set this vector to be perpendicular to a plane defined by 3 points
		// returns false if points are coincident or colinear
		bool perpendicular(
			const vec3&, const vec3&, const vec3&
		);
	};

	//| binary op : componentwise


	inline vec3 operator + (const vec3& a, const vec3& b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a[i] + b[i]; }
		return c;
	}

	inline vec3 operator + (Real a, const vec3& b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a + b[i]; }
		return c;
	}

	inline vec3 operator + (const vec3& a, Real b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a[i] + b; }
		return c;
	}



	inline vec3 operator - (const vec3& a, const vec3& b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a[i] - b[i]; }
		return c;
	}

	inline vec3 operator - (Real a, const vec3& b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a - b[i]; }
		return c;
	}

	inline vec3 operator - (const vec3& a, Real b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a[i] - b; }
		return c;
	}



	inline vec3 operator * (const vec3& a, const vec3& b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a[i] * b[i]; }
		return c;
	}

	inline vec3 operator * (Real a, const vec3& b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a * b[i]; }
		return c;
	}

	inline vec3 operator * (const vec3& a, Real b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a[i] * b; }
		return c;
	}



	inline vec3 operator / (const vec3& a, const vec3& b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a[i] / b[i]; }
		return c;
	}

	inline vec3 operator / (Real a, const vec3& b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a / b[i]; }
		return c;
	}

	inline vec3 operator / (const vec3& a, Real b)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = a[i] / b; }
		return c;
	}

	//| cumulative op : componentwise


	inline vec3 operator += (vec3& a, const vec3& b)
	{
		return a = (a + b);
	}

	inline vec3 operator += (vec3& a, Real b)
	{
		return a = (a + b);
	}



	inline vec3 operator -= (vec3& a, const vec3& b)
	{
		return a = (a - b);
	}

	inline vec3 operator -= (vec3& a, Real b)
	{
		return a = (a - b);
	}



	inline vec3 operator *= (vec3& a, const vec3& b)
	{
		return a = (a * b);
	}

	inline vec3 operator *= (vec3& a, Real b)
	{
		return a = (a * b);
	}



	inline vec3 operator /= (vec3& a, const vec3& b)
	{
		return a = (a / b);
	}

	inline vec3 operator /= (vec3& a, Real b)
	{
		return a = (a / b);
	}



	//| logical op : componentwise


	inline int operator == (const vec3& a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] == b[i]; }
		return c;
	}

	inline int operator == (Real a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a == b[i]; }
		return c;
	}

	inline int operator == (const vec3& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] == b; }
		return c;
	}



	inline int operator < (const vec3& a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] < b[i]; }
		return c;
	}

	inline int operator < (Real a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a < b[i]; }
		return c;
	}

	inline int operator < (const vec3& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] < b; }
		return c;
	}



	inline int operator <= (const vec3& a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] <= b[i]; }
		return c;
	}

	inline int operator <= (Real a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a <= b[i]; }
		return c;
	}

	inline int operator <= (const vec3& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] <= b; }
		return c;
	}



	inline int operator > (const vec3& a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] > b[i]; }
		return c;
	}

	inline int operator > (Real a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a > b[i]; }
		return c;
	}

	inline int operator > (const vec3& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] > b; }
		return c;
	}



	inline int operator >= (const vec3& a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] >= b[i]; }
		return c;
	}

	inline int operator >= (Real a, const vec3& b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a >= b[i]; }
		return c;
	}

	inline int operator >= (const vec3& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 3; ++i) { c = c && a[i] >= b; }
		return c;
	}



	//| unary op : componentwise
	inline vec3 operator - (const vec3& a)
	{
		vec3 c;
		for (int i = 0; i < 3; ++i) { c[i] = -a[i]; }
		return c;
	}

	inline vec3 absolute(const vec3& val)
	{
		return vec3(fabs(val[0]), fabs(val[1]), fabs(val[2]));
	}



	//| R^n -> R
	inline Real sum(const vec3& a)
	{
		Real s = 0;

		s += a[1 - 1];

		s += a[2 - 1];

		s += a[3 - 1];

		return s;
	}

	inline Real inner(const vec3& a, const vec3& b)
	{
		vec3 tmp = a * b;
		return sum(tmp);
	}

	inline Real squaredNorm(const vec3& a) {
		return inner(a, a);
	}

	inline Real norm(const vec3& a)
	{
		return sqrt(inner(a, a));
	}

	inline vec3 unit(const vec3& a)
	{
		Real n = norm(a);
		if (n < zero_epsilon)
			return 0;
		else
			return a / n;
	}

	inline Real angle(const vec3& a, const vec3& b)
	{
		Real ang = inner(unit(a), unit(b));
		if (ang > 1 - epsilon)
			return 0;
		else if (ang < -1 + epsilon)
			return M_PI;
		else
			return acos(ang);
	}

	inline vec3 proj(const vec3& axis, const vec3& a)
	{
		vec3 u = unit(axis);
		return inner(a, u) * u;
	}

	void store(FILE* fp, const vec3& v);
	//int scan(FILE* fp, const vec3& v);

	int apx_equal(const vec3& a, const vec3& b);
	int apx_equal(const vec3& a, const vec3& b, Real eps);

	/**
	* @brief structure for 4-dimensional Real type vector and its arithmetic.
	*/
	struct __declspec(dllexport) vec4 {
		Real v[4];
		static const int n;

		inline vec4() { }
		inline vec4(Real a) { v[1 - 1] = a;  v[2 - 1] = a;  v[3 - 1] = a;  v[4 - 1] = a; }
		inline vec4(Real v_1, Real v_2, Real v_3, Real v_4) { v[1 - 1] = v_1;  v[2 - 1] = v_2;  v[3 - 1] = v_3;  v[4 - 1] = v_4; }
		inline vec4(const ivec4& a) { v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1];  v[4 - 1] = a[4 - 1]; }
		inline vec4(const vec4& a) { v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1];  v[4 - 1] = a[4 - 1]; }

		inline vec4& operator=(const vec4& a) { v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1];  v[4 - 1] = a[4 - 1]; return *this; }
		inline Real& operator[] (int i) { return v[i]; }
		inline const Real&  operator[] (int i) const { return v[i]; }
		inline Real& operator() (int i) { return v[i % 4]; }
		inline const Real&  operator() (int i) const { return v[i % 4]; }

		inline bool is_zero() const { return (v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0 && v[3] == 0.0); }
		inline bool is_tiny(Real tiny_tol = epsilon) const {
			return (fabs(v[0]) <= tiny_tol && fabs(v[1]) <= tiny_tol && fabs(v[2]) <= tiny_tol && fabs(v[3]) <= tiny_tol);
		}

		bool unit();
		Real length() const;
	};





	//| binary op : componentwise


	inline vec4 operator + (const vec4& a, const vec4& b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a[i] + b[i]; }
		return c;
	}

	inline vec4 operator + (Real a, const vec4& b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a + b[i]; }
		return c;
	}

	inline vec4 operator + (const vec4& a, Real b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a[i] + b; }
		return c;
	}



	inline vec4 operator - (const vec4& a, const vec4& b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a[i] - b[i]; }
		return c;
	}

	inline vec4 operator - (Real a, const vec4& b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a - b[i]; }
		return c;
	}

	inline vec4 operator - (const vec4& a, Real b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a[i] - b; }
		return c;
	}



	inline vec4 operator * (const vec4& a, const vec4& b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a[i] * b[i]; }
		return c;
	}

	inline vec4 operator * (Real a, const vec4& b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a * b[i]; }
		return c;
	}

	inline vec4 operator * (const vec4& a, Real b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a[i] * b; }
		return c;
	}



	inline vec4 operator / (const vec4& a, const vec4& b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a[i] / b[i]; }
		return c;
	}

	inline vec4 operator / (Real a, const vec4& b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a / b[i]; }
		return c;
	}

	inline vec4 operator / (const vec4& a, Real b)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = a[i] / b; }
		return c;
	}



	//| cumulative op : componentwise


	inline vec4 operator += (vec4& a, const vec4& b)
	{
		return a = (a + b);
	}

	inline vec4 operator += (vec4& a, Real b)
	{
		return a = (a + b);
	}



	inline vec4 operator -= (vec4& a, const vec4& b)
	{
		return a = (a - b);
	}

	inline vec4 operator -= (vec4& a, Real b)
	{
		return a = (a - b);
	}



	inline vec4 operator *= (vec4& a, const vec4& b)
	{
		return a = (a * b);
	}

	inline vec4 operator *= (vec4& a, Real b)
	{
		return a = (a * b);
	}



	inline vec4 operator /= (vec4& a, const vec4& b)
	{
		return a = (a / b);
	}

	inline vec4 operator /= (vec4& a, Real b)
	{
		return a = (a / b);
	}



	//| logical op : componentwise


	inline int operator == (const vec4& a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] == b[i]; }
		return c;
	}

	inline int operator == (Real a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a == b[i]; }
		return c;
	}

	inline int operator == (const vec4& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] == b; }
		return c;
	}



	inline int operator < (const vec4& a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] < b[i]; }
		return c;
	}

	inline int operator < (Real a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a < b[i]; }
		return c;
	}

	inline int operator < (const vec4& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] < b; }
		return c;
	}



	inline int operator <= (const vec4& a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] <= b[i]; }
		return c;
	}

	inline int operator <= (Real a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a <= b[i]; }
		return c;
	}

	inline int operator <= (const vec4& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] <= b; }
		return c;
	}



	inline int operator > (const vec4& a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] > b[i]; }
		return c;
	}

	inline int operator > (Real a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a > b[i]; }
		return c;
	}

	inline int operator > (const vec4& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] > b; }
		return c;
	}



	inline int operator >= (const vec4& a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] >= b[i]; }
		return c;
	}

	inline int operator >= (Real a, const vec4& b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a >= b[i]; }
		return c;
	}

	inline int operator >= (const vec4& a, Real b)
	{
		int c = 1;
		for (int i = 0; i < 4; ++i) { c = c && a[i] >= b; }
		return c;
	}



	//| unary op : componentwise
	inline vec4 operator - (const vec4& a)
	{
		vec4 c;
		for (int i = 0; i < 4; ++i) { c[i] = -a[i]; }
		return c;
	}

	inline vec4 absolute(const vec4& val)
	{
		return vec4(fabs(val[0]), fabs(val[1]), fabs(val[2]), fabs(val[3]));
	}


	//| R^n -> R
	inline Real sum(const vec4& a)
	{
		Real s = 0;

		s += a[1 - 1];

		s += a[2 - 1];

		s += a[3 - 1];

		s += a[4 - 1];

		return s;
	}

	inline Real inner(const vec4& a, const vec4& b)
	{
		vec4 tmp = a * b;
		return sum(tmp);
	}
	inline Real squaredNorm(const vec4& a) {
		return inner(a, a);
	}
	inline Real norm(const vec4& a)
	{
		return sqrt(inner(a, a));
	}

	inline vec4 unit(const vec4& a)
	{
		Real n = norm(a);
		if (n < epsilon)
			return 0;
		else
			return a / n;
	}

	inline Real angle(const vec4& a, const vec4& b)
	{
		Real ang = inner(unit(a), unit(b));
		if (ang > 1 - epsilon)
			return 0;
		else if (ang < -1 + epsilon)
			return M_PI;
		else
			return acos(ang);
	}

	inline vec4 proj(const vec4& axis, const vec4& a)
	{
		vec4 u = unit(axis);
		return inner(a, u) * u;
	}

	void store(FILE* fp, const vec4& v);

	//int scan(FILE* fp, const vec4& v);

	int apx_equal(const vec4& a, const vec4& b);
	int apx_equal(const vec4& a, const vec4& b, Real eps);

	vec3 cross(const vec3& a, const vec3& b);


}; //namespace oph

#endif // !__vec_h
