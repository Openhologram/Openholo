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
	struct _declspec(dllexport) TwoDimMatrix
	{
	public:
		using typeT = typename std::enable_if<
			std::is_same<real, T>::value || std::is_same<real_t, T>::value ||
			std::is_same<int, T>::value ||
			std::is_same<uchar, T>::value ||
			std::is_same<Complex<real>, T>::value || std::is_same<Complex<real_t>, T>::value, T>::type;

		std::vector<T>* mat;
		ivec2 size;

		TwoDimMatrix(void) : size(1, 1) {
			init();
		}

		TwoDimMatrix(int x, int y) : size(x, y) {
			init();
		}

		TwoDimMatrix(ivec2 _size) : size(_size) {
			init();
		}

		TwoDimMatrix(TwoDimMatrix<T>& ref) : size(ref.size) {
			init();
			for (int x = 0; x < size[0]; x++)
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] = ref[x][y];
				}
		}

		~TwoDimMatrix() {
			release();
		}

		void init(void) {
			mat = new std::vector<T>[size[0]];
			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
					if (x == y && size[0] == size[1])
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

		TwoDimMatrix<T>& resize(int x, int y) {
			release();

			size[0] = x; size[1] = y;

			init();

			return *this;
		}	
		
		TwoDimMatrix<T>& identity(void) {
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

		T determinant(TwoDimMatrix<T>& _mat, int _size) {
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

		T cofactor(TwoDimMatrix<T>& _mat, int p, int q, int _size) {
			int i = 0, j = 0;
			int x = 0, y = 0;
			TwoDimMatrix<T> cmat(_size - 1, _size - 1);
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

		TwoDimMatrix<T>& inverse(void) {
			if (size[0] != size[1]) return *this;
			if (determinant() == 0) return *this;

			TwoDimMatrix<T> inv(size);
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

		TwoDimMatrix<T>& add(TwoDimMatrix<T>& p) {
			if (size != p.size) return *this;

			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] += p[x][y];
				}
			}

			return *this;
		}

		TwoDimMatrix<T>& sub(TwoDimMatrix<T>& p) {
			if (size != p.size) return *this;

			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] -= p[x][y];
				}
			}

			return *this;
		}

		TwoDimMatrix<T>& mul(TwoDimMatrix<T>& p) {
			if (size[0] != p.size[1]) return *this;

			TwoDimMatrix<T> res(p.size[1], size[0]);

			for (int x = 0; x < res.size[0]; x++) {
				for (int y = 0; y < res.size[1]; y++) {
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

		TwoDimMatrix<T>& div(TwoDimMatrix<T>& p) {
			if (size != p.size) return *this;

			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
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

		inline bool operator =(TwoDimMatrix<T>& p) {
			if (size != p.size)
				return false;

			for (int x = 0; x < size[0]; x++)
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] = p[x][y];
				}

			return true;
		}

		inline void operator =(T* p) {
			for (int x = 0; x < size[_X]; x++)
				for (int y = 0; y < size[_Y]; y++) {
					mat[x][y] = *p;
					p++;
				}
		}

		TwoDimMatrix<T>& operator ()(T args...) {
			va_list ap;

			__va_start(&ap, args);

			for (int x = 0; x < size[0]; x++)
				for (int y = 0; y < size[1]; y++) {
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

		const TwoDimMatrix<T>& operator +(const TwoDimMatrix<T>& p) {
			return *add(p);
		}

		const TwoDimMatrix<T>& operator -(const TwoDimMatrix<T>& p) {
			return *sub(p);
		}

		const TwoDimMatrix<T>& operator *(const TwoDimMatrix<T>& p) {
			return *mul(p);
		}

		const TwoDimMatrix<T>& operator /(const TwoDimMatrix<T>& p) {
			return *div(p);
		}

		const TwoDimMatrix<T>& operator +(const T& p) {
			for (int x = 0; x < size[0]; x++)
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] += p;
				}
			return *this;
		}

		const TwoDimMatrix<T>& operator -(const T& p) {
			for (int x = 0; x < size[0]; x++)
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] -= p;
				}
			return *this;
		}

		const TwoDimMatrix<T>& operator *(const T& p) {
			for (int x = 0; x < size[0]; x++)
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] *= p;
				}
			return *this;
		}

		const TwoDimMatrix<T>& operator /(const T& p) {
			for (int x = 0; x < size[0]; x++)
				for (int y = 0; y < size[1]; y++) {
					mat[x][y] /= p;
				}
			return *this;
		}

		//print test
		void Print(const char* _context) {
			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
					printf(_context, mat[x][y]);
				}
				cout << endl;
			}
			cout << endl;
		}
	};

	typedef oph::TwoDimMatrix<int> OphIntField;
	typedef oph::TwoDimMatrix<uchar> OphByteField;
	typedef oph::TwoDimMatrix<real> OphRealField;
	typedef oph::TwoDimMatrix<real_t> OphRealTField;
	typedef oph::TwoDimMatrix<Complex<real>> OphComplexField;
	typedef oph::TwoDimMatrix<Complex<real_t>> OphComplexTField;

	typedef OphComplexField Mat;
	typedef OphComplexField MatF;
}

#endif // !__mat_h