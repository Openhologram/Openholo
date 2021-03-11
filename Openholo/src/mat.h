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

#ifndef __mat_h
#define __mat_h

#include "ivec.h"
#include "typedef.h"
#include "complex.h"
#include "define.h"

#include <vector>

using namespace oph;

namespace oph
{
	template<typename T>
	class _declspec(dllexport) matrix
	{
	public:
		using typeT = typename std::enable_if<
			std::is_same<Real, T>::value || std::is_same<Real_t, T>::value ||
			std::is_same<int, T>::value ||
			std::is_same<uchar, T>::value ||
			std::is_same<Complex<Real>, T>::value || std::is_same<Complex<Real_t>, T>::value, T>::type;

		std::vector<T>* mat;
		ivec2 size;

		matrix(void) : size(1, 1) {
			init();
		}

		matrix(int x, int y) : size(x, y) {
			init();
		}

		matrix(ivec2 _size) : size(_size) {
			init();
		}

		matrix(const matrix<T>& ref) : size(ref.size) {
			init();
			for (int x = 0; x < size[_X]; x++)
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] = ref.mat[x][y];
				}
		}

		~matrix() {
			release();
		}

		void init(void) {
			mat = new std::vector<T>[size[0]];
			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					if (x == y && size[_X] == size[_Y])
						mat[x].push_back(1);
					else
						mat[x].push_back(0);
				}
			}
		}

		void release(void) {
			if (!mat) return;
			delete[] mat;
			mat = nullptr;
		}

		oph::ivec2& getSize(void) { return size; }

		matrix<T>& resize(int x, int y) {
			release();

			size[0] = x; size[1] = y;

			init();

			return *this;
		}

		matrix<T>& identity(void) {
			if (size[_X] != size[_Y]) return *this;
			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					if (x == y)
						mat[x][y] = 1;
					else
						mat[x][y] = 0;
				}
			}
			return *this;
		}

		matrix<T>& zeros(void) {
			for (int col = 0; col < size[_COL]; col++) {
				for (int row = 0; row < size[_ROW]; row++) {
					mat[col][row] = 0;
				}
			}
			return *this;
		}

		//T determinant(void) {
		//	if (size[_X] != size[_Y]) return 0;

		//	return determinant(*this, size[_X]);
		//}

		//T determinant(matrix<T>& _mat, int _size) {
		//	int p = 0, q = 0;
		//	T det = 0;

		//	if (_size == 1)	return _mat[0][0];
		//	else if (_size == 2) return _mat[0][0] * _mat[1][1] - _mat[0][1] * _mat[1][0];
		//	else {
		//		for (q = 0, det = 0; q<_size; q++) {
		//			det = det + _mat[0][q] * cofactor(_mat, 0, q, _size);
		//		}
		//		return det;
		//	}
		//	return 0;
		//}

		//T cofactor(matrix<T>& _mat, int p, int q, int _size) {
		//	int i = 0, j = 0;
		//	int x = 0, y = 0;
		//	matrix<T> cmat(_size - 1, _size - 1);
		//	T cofactor = 0;

		//	for (i = 0, x = 0; i<_size; i++) {
		//		if (i != p) {
		//			for (j = 0, y = 0; j<_size; j++) {
		//				if (j != q) {
		//					cmat[x][y] = _mat[i][j];
		//					y++;
		//				}
		//			}
		//			x++;
		//		}
		//	}

		//	cofactor = pow(-1, p)*pow(-1, q)*determinant(cmat, _size - 1);
		//	return cofactor;
		//}

		//void swapRow(int i, int j) {
		//	if (i == j) return;
		//	for (int k = 0; k < size[0]; k++) swap(mat[i][k], mat[j][k]);
		//}

		//matrix<T>& inverse(void) {
		//	if (size[_X] != size[_Y]) return *this;
		//	if (determinant() == 0) return *this;

		//	matrix<T> inv(size);
		//	inv.identity();

		//	for (int k = 0; k < size[0]; k++) {
		//		int t = k - 1;

		//		while (t + 1 < size[0] && !mat[++t][k]);
		//		if (t == size[0] - 1 && !mat[t][k]) return *this;
		//		swapRow(k, t), inv.swapRow(k, t);

		//		T d = mat[k][k];
		//		for (int j = 0; j < size[0]; j++)
		//			mat[k][j] /= d, inv[k][j] /= d;


		//		for (int i = 0; i < size[0]; i++)
		//			if (i != k) {
		//				T m = mat[i][k];
		//				for (int j = 0; j < size[0]; j++) {
		//					if (j >= k) mat[i][j] -= mat[k][j] * m;
		//					inv[i][j] -= inv[k][j] * m;
		//				}
		//			}
		//	}

		//	*this = inv;

		//	return *this;
		//}

		matrix<T>& add(matrix<T>& p) {
			if (size != p.size) return *this;

			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] += p[x][y];
				}
			}

			return *this;
		}

		matrix<T>& sub(matrix<T>& p) {
			if (size != p.size) return *this;

			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] -= p[x][y];
				}
			}

			return *this;
		}

		matrix<T>& mul(matrix<T>& p) {
			if (size[_X] != p.size[_Y]) return *this;

			matrix<T> res(p.size[_Y], size[_X]);

			for (int x = 0; x < res.size[_X]; x++) {
				for (int y = 0; y < res.size[_Y]; y++) {
					res[x][y] = 0;
					for (int num = 0; num < p.size[_X]; num++) {
						res[x][y] += mat[x][num] * p[num][y];
					}
				}
			}
			this->resize(res.size[_X], res.size[_Y]);
			*this = res;

			return *this;
		}

		matrix<T>& div(matrix<T>& p) {
			if (size != p.size) return *this;

			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					if (p[x][y] == 0) continue;
					mat[x][y] /= p[x][y];
				}
			}

			return *this;
		}

		matrix<T>& mulElem(matrix<T>& p) {
			if (size != p.size) return *this;

			matrix<T> res(size);

			for (int x = 0; x < res.size[_X]; x++) {
				for (int y = 0; y < res.size[_Y]; y++) {
					res[x][y] = 0;
					res[x][y] = mat[x][y] * p.mat[x][y];
				}
			}

			*this = res;

			return *this;
		}


		std::vector<T>& operator[](const int index) {
			return mat[index];
		}

		T& operator ()(int x, int y) {
			return mat[x][y];
		}

		inline void operator =(matrix<T>& p) {
			if (size != p.size)
				return;

			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] = p[x][y];
				}
			}
		}

		inline void operator =(T* p) {
			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] = *p;
					p++;
				}
			}
		}

		//matrix<T>& operator ()(T args...) {
		//	va_list ap;

		//	__va_start(&ap, args);

		//	for (int x = 0; x < size[_X]; x++) {
		//		for (int y = 0; y < size[_Y]; y++) {
		//			if (x == 0 && y == 0) {
		//				mat[x][y] = args;
		//				continue;
		//			}
		//			T n = __crt_va_arg(ap, T);
		//			mat[x][y] = n;
		//		}
		//	}
		//	__crt_va_end(ap);

		//	return *this;
		//}

		matrix<T>& operator +(matrix<T>& p) {
			return add(p);
		}

		matrix<T>& operator -(matrix<T>& p) {
			return sub(p);
		}

		matrix<T>& operator *(matrix<T>& p) {
			return mul(p);
		}

		matrix<T>& operator /(matrix<T>& p) {
			return div(p);
		}

		matrix<T>& operator +(const T& p) {
			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] += p;
				}
			}
			return *this;
		}

		const matrix<T>& operator -(const T& p) {
			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] -= p;
				}
			}
			return *this;
		}

		const matrix<T>& operator *(const T& p) {
			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] *= p;
				}
			}
			return *this;
		}

		const matrix<T>& operator /(const T& p) {
			for (int x = 0; x < size[_X]; x++) {
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] /= p;
				}
			}
			return *this;
		}

		//print test
		//void Print(const char* _context) {
		//	for (int x = 0; x < size[_X]; x++) {
		//		for (int y = 0; y < size[_Y]; y++) {
		//			printf(_context, mat[x][y]);
		//		}
		//		cout << endl;
		//	}
		//	cout << endl;
		//}
	};

	typedef oph::matrix<int> OphIntField;
	typedef oph::matrix<uchar> OphByteField;
	typedef oph::matrix<Real> OphRealField;
	typedef oph::matrix<Real_t> OphRealTField;
	typedef oph::matrix<Complex<Real>> OphComplexField;
	typedef oph::matrix<Complex<Real_t>> OphComplexTField;

	typedef OphComplexField Mat;
	typedef OphComplexTField MatF;
}

#endif // !__mat_h