#ifndef __function_h
#define __function_h

#include "complex.h"
#include "real.h"

namespace oph
{
	template<typename type, typename T>
	inline type force_cast(const Complex<T>& p) {
		return type(p.re);
	}

	template<typename type, typename T>
	inline type force_cast(const T& p) {
		return type(p);
	}

	inline real& minOfArr(const std::vector<real>& arr) {
		real min = _MAXDOUBLE;
		for (auto item : arr) { if (item < min) min = item; }
		return min;
	}

	inline real& maxOfArr(const std::vector<real>& arr) {
		real max = _MINDOUBLE;
		for (auto item : arr) { if (item > max) max = item; }
		return max;
	}

	inline void absArr(const std::vector<real>& src, std::vector<real>& dst) {
		dst.clear();
		dst.reserve(src.size());
		for (auto item : src) { dst.push_back(abs(item)); }
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
				oph::uchar *res_pos = dst + xdx + (ny - ydx - 1)*nx;
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
		dst->clear();
		dst->reserve(src->size());

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

	/**
	* @brief Shifts the elements by shift_x, shift_y.
	*/
	template<typename T>
	inline void circshift(const T* src, T* dst, int shift_x, int shift_y, int xdim, int ydim) {
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
}

#endif // !__function_h
