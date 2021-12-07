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

#ifndef __function_h
#define __function_h

#include "complex.h"
#include "mat.h"
#include "vec.h"

#include <chrono>
#include <random>

namespace oph
{
#define CUR_TIME std::chrono::system_clock::now()
#define ELAPSED_TIME(x,y) ((std::chrono::duration<double>)(y - x)).count()
#define CUR_TIME_DURATION CUR_TIME.time_since_epoch()
#define CUR_TIME_DURATION_MILLI_SEC std::chrono::duration_cast<std::chrono::milliseconds>(CUR_TIME_DURATION).count()

	template<typename type, typename T>
	inline type force_cast(const Complex<T>& p) {
		return type(p._Val[_RE]);
	}

	template<typename type, typename T>
	inline type force_cast(const T& p) {
		return type(p);
	}

	inline Real minOfArr(const std::vector<Real>& arr) {
		Real min = MAX_DOUBLE;
		for (auto item : arr) { if (item < min) min = item; }
		return min;
	}

	inline Real minOfArr(const Real* src, const int& size) {
		Real min = MAX_DOUBLE;
		for (int i = 0; i < size; i++) {
			if (*(src + i) < min) min = *(src + i);
		}
		return min;
	}

	inline Real maxOfArr(const std::vector<Real>& arr) {
		Real max = MIN_DOUBLE;
		for (auto item : arr) { if (item > max) max = item; }
		return max;
	}

	inline Real maxOfArr(const Real* src, const int& size) {
		Real max = MIN_DOUBLE;
		for (int i = 0; i < size; i++) {
			if (*(src + i) > max) max = *(src + i);
		}
		return max;
	}

	inline Real average(const Real* src, const int& size) {
		Real ave;
		Real sum = 0;
		for (int i = 0; i < size; i++) {
			sum += *(src + i);
		}
		ave = sum / size;

		return ave;
	}

	template<typename T>
	inline void abs(const oph::Complex<T>& src, oph::Complex<T>& dst) {
		dst = std::abs(force_cast<std::complex<Real>>(src));
	}

	template<typename T>
	inline void absArr(const std::vector<Complex<T>>& src, std::vector<oph::Complex<T>>& dst) {
		dst.clear();
		dst.reserve(src.size());
		for (auto item : src) dst.push_back(oph::abs(item));
	}

	template<typename T>
	inline void absMat(const oph::matrix<oph::Complex<T>>& src, oph::matrix<oph::Complex<T>>& dst) {
		if (src.getSize() != dst.getSize()) return;
		oph::ivec2 matSize;
		for (int x = 0; x < matSize[_X]; x++) {
			for (int y = 0; y < matSize[_Y]; y++)
				oph::abs(src[x][y], dst[x][y]);
		}
	}

	template<typename T>
	inline void absCplx(const oph::Complex<T>& src, T& dst) {
		dst = sqrt(src._Val[_RE]*src._Val[_RE] + src._Val[_IM]*src._Val[_IM]);
	}

	template<typename T>
	inline void absCplxArr(const oph::Complex<T>* src, T* dst, const int& size) {
		for (int i = 0; i < size; i++) {
			absCplx<T>(*(src + i), *(dst + i));
		}
	}

	template<typename T>
	inline T getReal(const oph::Complex<T> src) {
		return src->_Val[_RE];
	}

	template<typename T>
	inline void realPart(const oph::Complex<T>* src, T* dst, const int& size) {
		for (int i = 0; i < size; i++) {
			*(dst + i) = (src + i)->_Val[_RE];
		}
	}

	template<typename T>
	inline void angle(const std::vector<Complex<T>>& src, std::vector<T>& dst) {
		dst.clear();
		dst.reserve(src.size());
		for (auto item : src) {	dst.push_back(src.angle());	}
	}

	template<typename T>
	inline T angle(const oph::Complex<T>& src) {
		T dst;
		dst = atan2(src.imag(), src.real());
		return dst;
	}

	/**
	* @brief Normalize all elements of Complex<T>* src from 0 to 1.
	*/
	template<typename T>
	inline void normalize(const Complex<T>* src, Complex<T>* dst, const int& size) {
		Real* abs = new Real[size];
		oph::absCplxArr<Real>(src, abs, size);

		Real max = oph::maxOfArr(abs, size);
		
		for (int i = 0; i < size; i++) {
			*(dst + i) = *(src + i) / max;
		}
		delete[] abs;
	}
	template<typename T>
	inline void normalize(const T* src, T* dst, const int& size) {
		
		Real min = oph::minOfArr(src, size);
		Real max = oph::maxOfArr(src, size);

		for (int i = 0; i < size; i++) {
			*(dst + i) = (*(src + i) - min) / (max - min);
		}
	}

	/**
	* @brief Normalize all elements of T* src from 0 to 255.
	*/
	template<typename T>
	inline void normalize(T* src, oph::uchar* dst, const oph::uint nx, const oph::uint ny) {
		T minVal, maxVal;
		for (oph::uint ydx = 0; ydx < ny; ydx++){
			for (oph::uint xdx = 0; xdx < nx; xdx++){
				T *temp_pos = src + xdx + ydx * nx;
				if ((xdx == 0) && (ydx == 0)) {	minVal = *(temp_pos); maxVal = *(temp_pos);	}
				else {
					if ((*temp_pos) < minVal) minVal = (*temp_pos);
					if ((*temp_pos) > maxVal) maxVal = (*temp_pos);
				}
			}
		}
		
		for (oph::uint ydx = 0; ydx < ny; ydx++) {
			for (oph::uint xdx = 0; xdx < nx; xdx++) {
				T *src_pos = src + xdx + ydx * nx;
				//oph::uchar *res_pos = dst + xdx + (ny - ydx - 1)*nx;
				oph::uchar *res_pos = dst + xdx + ydx * nx;
				*(res_pos) = oph::force_cast<oph::uchar>(((*(src_pos)-minVal) / (maxVal - minVal)) * 255 + 0.5);
			}
		}
	}

	/**
	* @brief Normalize all elements from 0 to 255. 
	*/
	template<typename T>
	inline void normalize(const std::vector<T>* src, std::vector<oph::uchar>* dst) {
		T minVal, maxVal;
		if (src->size() != dst->size())
		{
			dst->clear();
			dst->reserve(src->size());
		}

		auto iter = src->begin();
		for (iter; iter != src->end(); iter++) {
			if (iter == src->begin()) {
				minVal = *iter;
				maxVal = *iter;
			} else {
				if (*iter < minVal) minVal = *iter;
				if (*iter > maxVal) maxVal = *iter;
			}
		}

		iter = src->begin();
		for (iter; iter != src->end(); iter++)
			dst->push_back(oph::force_cast<oph::uchar>((((*iter) - minVal) / (maxVal - minVal)) * 255 + 0.5));
	}

	/**
	* @brief Set elements to specific values from begin index to end index.
	*/
	template<typename T>
	inline void memsetArr(const std::vector<T>* pArr, T _Value, oph::uint beginIndex, oph::uint endIndex){
		auto iter = pArr->begin() + (beginIndex);
		auto it_end = pArr->begin() + (endIndex);
		for (; iter != it_end; iter++) { (*iter) = _Value; }
	}

	template<typename T>
	void memsetArr(T* pArr, const T& _Value, const oph::uint& beginIndex, const oph::uint& endIndex) {
		for (uint i = beginIndex; i <= endIndex; i++) {
			*(pArr + i) = _Value;
		}
	}

	/**
	* @brief Shifts the elements by shift_x, shift_y.
	*/
	template<typename T>
	inline void circShift(const T* src, T* dst, int shift_x, int shift_y, int xdim, int ydim) {
		for (int i = 0; i < xdim; i++) {
			int ti = (i + shift_x) % xdim;
			if (ti < 0) ti = xdim + ti;
			for (int j = 0; j < ydim; j++) {
				int tj = (j + shift_y) % ydim;
				if (tj < 0) tj = ydim + tj;
				dst[ti * ydim + tj] = src[i * ydim + j];
			}
		}
	}

	/**
	* @brief Get random Real value from min to max 
	* @param[in] min minimum value.
	* @param[in] max maximum value.
	* @param[in] _SEED_VALUE : Random seed value can be used to create a specific random number pattern@n
	*					   If the seed values are the same, random numbers of the same pattern are always output.
	* @return Type: <B>Real</B>\n
	*				The return value is <B>random value</B>.
	*/
	inline Real rand(const Real min, const Real max, oph::ulong _SEED_VALUE = 0) {
		std::mt19937_64 rand_dev;
		if (!_SEED_VALUE) rand_dev = std::mt19937_64(CUR_TIME_DURATION_MILLI_SEC);
		else rand_dev = std::mt19937_64(_SEED_VALUE);

		std::uniform_real_distribution<Real> dist(min, max);

		return dist(rand_dev);
	}

	/**
	* @brief Get random Integer value from min to max
	* @param[in] min minimum value.
	* @param[in] max maximum value.
	* @param[in] _SEED_VALUE : Random seed value can be used to create a specific random number pattern@n
	*					   If the seed values are the same, random numbers of the same pattern are always output.
	* @return Type: <B>int</B>\n
	*				The return value is <B>random value</B>.
	*/
	inline int rand(const int min, const int max, oph::ulong _SEED_VALUE = 0) {
		std::mt19937_64 rand_dev;
		if (!_SEED_VALUE) rand_dev = std::mt19937_64(CUR_TIME_DURATION_MILLI_SEC);
		else rand_dev = std::mt19937_64(_SEED_VALUE);

		std::uniform_int_distribution<int> dist(min, max);

		return dist(rand_dev);
	}

	inline void getPhase(oph::Complex<Real>* src, Real* dst, const int& size)
	{
		for (int i = 0; i < size; i++) {
			*(dst + i) = angle<Real>(*(src + i)) + M_PI;
		}
	}

	inline void getAmplitude(oph::Complex<Real>* src, Real* dst, const int& size) {
		absCplxArr<Real>(src, dst, size);
	}

	template<typename T>
	inline void Field2Buffer(matrix<T>& src, T** dst) {
		ivec2 bufferSize = src.getSize();

		*dst = new oph::Complex<Real>[bufferSize[_X] * bufferSize[_Y]];

		int idx = 0;
		for (int x = 0; x < bufferSize[_X]; x++) {
			for (int y = 0; y < bufferSize[_Y]; y++) {
				(*dst)[idx] = src[x][y];
				idx++;
			}
		}
	}

	template<typename T>
	inline void Buffer2Field(const T* src, matrix<T>& dst, const ivec2 buffer_size) {
		ivec2 matSize = buffer_size;

		int idx = 0;
		for (int x = 0; x < matSize[_X]; x++) {
			for (int y = 0; y < matSize[_Y]; y++) {
				dst[x][y] = src[idx];
				idx++;
			}
		}
	}


	/**
	* @brief
	*/
	inline void meshGrid() {

	}
}


#endif // !__function_h
