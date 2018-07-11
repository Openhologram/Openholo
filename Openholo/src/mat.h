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

		matrix(matrix<T>& ref) : size(ref.size) {
			init();
			for (int y = 0; y < size[_Y]; y++)
				for (int x = 0; x < size[_X]; x++) {
					mat[x][y] = ref[x][y];
				}
		}

		~matrix() {
			release();
		}

		void init(void) {
			mat = new std::vector<T>[size[0]];
			for (int y = 0; y < size[_Y]; y++) {
				for (int x = 0; x < size[_X]; x++) {
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
			if (size[0] != size[0]) return this;
			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
					if (x == y)
						mat[x][y] = 1;
					else
						mat[x][y] = 0;
				}
			}
			return *this;
		}

		T determinant(void) {
			if (size[0] != size[1]) return 0;

			return determinant(*this, size[0]);
		}

		T determinant(matrix<T>& _mat, int _size) {
			int p = 0, q = 0;
			T det = 0;

			if (_size == 1)	return _mat[0][0];
			else if (_size == 2) return _mat[0][0] * _mat[1][1] - _mat[0][1] * _mat[1][0];
			else {
				for (q = 0, det = 0; q<_size; q++) {
					det = det + _mat[0][q] * cofactor(_mat, 0, q, _size);
				}
				return det;
			}
			return 0;
		}

		T cofactor(matrix<T>& _mat, int p, int q, int _size) {
			int i = 0, j = 0;
			int x = 0, y = 0;
			matrix<T> cmat(_size - 1, _size - 1);
			T cofactor = 0;

			for (i = 0, x = 0; i<_size; i++) {
				if (i != p) {
					for (j = 0, y = 0; j<_size; j++) {
						if (j != q) {
							cmat[x][y] = _mat[i][j];
							y++;
						}
					}
					x++;
				}
			}

			cofactor = pow(-1, p)*pow(-1, q)*determinant(cmat, _size - 1);
			return cofactor;
		}

		void swapRow(int i, int j) {
			if (i == j) return;
			for (int k = 0; k < size[0]; k++) swap(mat[i][k], mat[j][k]);
		}

		matrix<T>& inverse(void) {
			if (size[0] != size[1]) return *this;
			if (determinant() == 0) return *this;

			matrix<T> inv(size);
			inv.identity();

			for (int k = 0; k < size[0]; k++) {
				int t = k - 1;

				while (t + 1 < size[0] && !mat[++t][k]);
				if (t == size[0] - 1 && !mat[t][k]) return *this;
				swapRow(k, t), inv.swapRow(k, t);

				T d = mat[k][k];
				for (int j = 0; j < size[0]; j++)
					mat[k][j] /= d, inv[k][j] /= d;


				for (int i = 0; i < size[0]; i++)
					if (i != k) {
						T m = mat[i][k];
						for (int j = 0; j < size[0]; j++) {
							if (j >= k) mat[i][j] -= mat[k][j] * m;
							inv[i][j] -= inv[k][j] * m;
						}
					}
			}

			*this = inv;

			return *this;
		}

		matrix<T>& add(matrix<T>& p) {
			if (size != p.size) return *this;

			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] += p[x][y];
				}
			}

			return *this;
		}

		matrix<T>& sub(matrix<T>& p) {
			if (size != p.size) return *this;

			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] -= p[x][y];
				}
			}

			return *this;
		}

		matrix<T>& mul(matrix<T>& p) {
			if (size[0] != p.size[1]) return *this;

			matrix<T> res(p.size[1], size[0]);

			for (int y = 0; y < res.size[0]; y++) {
				for (int x = 0; x < res.size[1]; x++) {
					res[x][y] = 0;
					for (int num = 0; num < size[1]; num++)	{
						res[x][y] += mat[x][num] * p[num][y];
					}
				}
			}

			this->resize(res.size[0], res.size[1]);
			*this = res;

			return *this;
		}

		matrix<T>& div(matrix<T>& p) {
			if (size != p.size) return *this;

			for (int y = 0; y < size[_Y]; y++) {
				for (int x = 0; x < size[_X]; x++) {
					mat[x][y] /= p[x][y];
				}
			}

			return *this;
		}


		std::vector<T>& operator[](const int index) {
			return mat[index];
		}

		T& operator ()(int x, int y) {
			return mat[x][y];
		}

		inline bool operator =(matrix<T>& p) {
			if (size != p.size)
				return false;

			for (int y = 0; y < size[_Y]; y++)
				for (int x = 0; x < size[_X]; x++) {
					mat[x][y] = p[x][y];
				}

			return true;
		}

		inline void operator =(T* p) {
			for (int y = 0; y < size[_Y]; y++)
				for (int x = 0; x < size[_X]; x++) {
					mat[x][y] = *p;
					p++;
				}
		}

		matrix<T>& operator ()(T args...) {
			va_list ap;

			__va_start(&ap, args);

			for (int y = 0; y < size[_Y]; y++)
				for (int x = 0; x < size[X]; x++) {
					if (x == 0 && y == 0) {
						mat[x][y] = args;
						continue;
					}
					T n = __crt_va_arg(ap, T);
					mat[x][y] = n;
				}
			__crt_va_end(ap);

			return *this;
		}

		const matrix<T>& operator +(const matrix<T>& p) {
			return *add(p);
		}

		const matrix<T>& operator -(const matrix<T>& p) {
			return *sub(p);
		}

		const matrix<T>& operator *(const matrix<T>& p) {
			return *mul(p);
		}

		const matrix<T>& operator /(const matrix<T>& p) {
			return *div(p);
		}

		const matrix<T>& operator +(const T& p) {
			for (int y = 0; y < size[_Y]; y++)
				for (int x = 0; x < size[_X]; x++) {
					mat[x][y] += p;
				}
			return *this;
		}

		const matrix<T>& operator -(const T& p) {
			for (int y = 0; y < size[_Y]; y++)
				for (int x = 0; x < size[_X]; x++) {
					mat[x][y] -= p;
				}
			return *this;
		}

		const matrix<T>& operator *(const T& p) {
			for (int y = 0; y < size[_Y]; y++)
				for (int x = 0; x < size[_X]; x++) {
					mat[x][y] *= p;
				}
			return *this;
		}

		const matrix<T>& operator /(const T& p) {
			for (int y = 0; y < size[_Y]; y++)
				for (int x = 0; x < size[_X]; x++) {
					mat[x][y] /= p;
				}
			return *this;
		}

		//print test
		void Print(const char* _context) {
			for (int y = 0; y < size[_Y]; y++) {
				for (int x = 0; x < size[_X]; x++) {
					printf(_context, mat[x][y]);
				}
				cout << endl;
			}
			cout << endl;
		}
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