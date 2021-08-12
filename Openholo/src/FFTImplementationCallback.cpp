//
//  FFTImplementationCallback.cpp
//
//  Code generation for function 'FFTImplementationCallback'
//


// Include files
#include "FFTImplementationCallback.h"
#include "rt_nonfinite.h"
#include <cmath>

// Function Definitions
void FFTImplementationCallback::r2br_r2dit_trig_impl(const coder::array<creal_T,
	2U> &x, int xoffInit, int unsigned_nRows, const coder::array<double, 2U>
	&costab, const coder::array<double, 2U> &sintab, coder::array<creal_T, 1U> &y)
{
	int ix;
	int iy;
	int iheight;
	int istart;
	int nRowsD2;
	int k;
	int ju;
	int i;
	double temp_re;
	double temp_im;
	double twid_re;
	double twid_im;
	y.set_size(unsigned_nRows);
	if (unsigned_nRows > x.size(0)) {
		y.set_size(unsigned_nRows);
		for (iy = 0; iy < unsigned_nRows; iy++) {
			y[iy].re = 0.0;
			y[iy].im = 0.0;
		}
	}

	ix = xoffInit;
	iheight = x.size(0);
	if (iheight >= unsigned_nRows) {
		iheight = unsigned_nRows;
	}

	istart = unsigned_nRows - 2;
	nRowsD2 = unsigned_nRows / 2;
	k = nRowsD2 / 2;
	iy = 0;
	ju = 0;
	for (i = 0; i <= iheight - 2; i++) {
		boolean_T tst;
		y[iy] = x[ix];
		iy = unsigned_nRows;
		tst = true;
		while (tst) {
			iy >>= 1;
			ju ^= iy;
			tst = ((ju & iy) == 0);
		}

		iy = ju;
		ix++;
	}

	y[iy] = x[ix];
	if (unsigned_nRows > 1) {
		for (i = 0; i <= istart; i += 2) {
			temp_re = y[i + 1].re;
			temp_im = y[i + 1].im;
			twid_re = y[i].re;
			twid_im = y[i].im;
			y[i + 1].re = y[i].re - y[i + 1].re;
			y[i + 1].im = y[i].im - y[i + 1].im;
			twid_re += temp_re;
			twid_im += temp_im;
			y[i].re = twid_re;
			y[i].im = twid_im;
		}
	}

	iy = 2;
	ix = 4;
	iheight = ((k - 1) << 2) + 1;
	while (k > 0) {
		int temp_re_tmp;
		for (i = 0; i < iheight; i += ix) {
			temp_re_tmp = i + iy;
			temp_re = y[temp_re_tmp].re;
			temp_im = y[temp_re_tmp].im;
			y[temp_re_tmp].re = y[i].re - y[temp_re_tmp].re;
			y[temp_re_tmp].im = y[i].im - y[temp_re_tmp].im;
			y[i].re = y[i].re + temp_re;
			y[i].im = y[i].im + temp_im;
		}

		istart = 1;
		for (ju = k; ju < nRowsD2; ju += k) {
			int ihi;
			twid_re = costab[ju];
			twid_im = sintab[ju];
			i = istart;
			ihi = istart + iheight;
			while (i < ihi) {
				temp_re_tmp = i + iy;
				temp_re = twid_re * y[temp_re_tmp].re - twid_im * y[temp_re_tmp].im;
				temp_im = twid_re * y[temp_re_tmp].im + twid_im * y[temp_re_tmp].re;
				y[temp_re_tmp].re = y[i].re - temp_re;
				y[temp_re_tmp].im = y[i].im - temp_im;
				y[i].re = y[i].re + temp_re;
				y[i].im = y[i].im + temp_im;
				i += ix;
			}

			istart++;
		}

		k /= 2;
		iy = ix;
		ix += ix;
		iheight -= iy;
	}
}

void FFTImplementationCallback::r2br_r2dit_trig_impl(const coder::array<creal_T,
	1U> &x, int unsigned_nRows, const coder::array<double, 2U> &costab, const
	coder::array<double, 2U> &sintab, coder::array<creal_T, 1U> &y)
{
	int iDelta2;
	int iy;
	int iheight;
	int nRowsD2;
	int k;
	int ix;
	int ju;
	int i;
	double temp_re;
	double temp_im;
	double twid_re;
	double twid_im;
	y.set_size(unsigned_nRows);
	if (unsigned_nRows > x.size(0)) {
		y.set_size(unsigned_nRows);
		for (iy = 0; iy < unsigned_nRows; iy++) {
			y[iy].re = 0.0;
			y[iy].im = 0.0;
		}
	}

	iDelta2 = x.size(0);
	if (iDelta2 >= unsigned_nRows) {
		iDelta2 = unsigned_nRows;
	}

	iheight = unsigned_nRows - 2;
	nRowsD2 = unsigned_nRows / 2;
	k = nRowsD2 / 2;
	ix = 0;
	iy = 0;
	ju = 0;
	for (i = 0; i <= iDelta2 - 2; i++) {
		boolean_T tst;
		y[iy] = x[ix];
		iy = unsigned_nRows;
		tst = true;
		while (tst) {
			iy >>= 1;
			ju ^= iy;
			tst = ((ju & iy) == 0);
		}

		iy = ju;
		ix++;
	}

	y[iy] = x[ix];
	if (unsigned_nRows > 1) {
		for (i = 0; i <= iheight; i += 2) {
			temp_re = y[i + 1].re;
			temp_im = y[i + 1].im;
			twid_re = y[i].re;
			twid_im = y[i].im;
			y[i + 1].re = y[i].re - y[i + 1].re;
			y[i + 1].im = y[i].im - y[i + 1].im;
			twid_re += temp_re;
			twid_im += temp_im;
			y[i].re = twid_re;
			y[i].im = twid_im;
		}
	}

	iy = 2;
	iDelta2 = 4;
	iheight = ((k - 1) << 2) + 1;
	while (k > 0) {
		int temp_re_tmp;
		for (i = 0; i < iheight; i += iDelta2) {
			temp_re_tmp = i + iy;
			temp_re = y[temp_re_tmp].re;
			temp_im = y[temp_re_tmp].im;
			y[temp_re_tmp].re = y[i].re - y[temp_re_tmp].re;
			y[temp_re_tmp].im = y[i].im - y[temp_re_tmp].im;
			y[i].re = y[i].re + temp_re;
			y[i].im = y[i].im + temp_im;
		}

		ix = 1;
		for (ju = k; ju < nRowsD2; ju += k) {
			int ihi;
			twid_re = costab[ju];
			twid_im = sintab[ju];
			i = ix;
			ihi = ix + iheight;
			while (i < ihi) {
				temp_re_tmp = i + iy;
				temp_re = twid_re * y[temp_re_tmp].re - twid_im * y[temp_re_tmp].im;
				temp_im = twid_re * y[temp_re_tmp].im + twid_im * y[temp_re_tmp].re;
				y[temp_re_tmp].re = y[i].re - temp_re;
				y[temp_re_tmp].im = y[i].im - temp_im;
				y[i].re = y[i].re + temp_re;
				y[i].im = y[i].im + temp_im;
				i += iDelta2;
			}

			ix++;
		}

		k /= 2;
		iy = iDelta2;
		iDelta2 += iDelta2;
		iheight -= iy;
	}
}

void FFTImplementationCallback::b_dobluesteinfft(const coder::array<creal_T, 2U>
	&x, int n2blue, int nfft, const coder::array<double, 2U> &costab, const coder::
	array<double, 2U> &sintab, const coder::array<double, 2U> &sintabinv, coder::
	array<creal_T, 2U> &y)
{
	int nInt2m1;
	coder::array<creal_T, 1U> wwc;
	int idx;
	int rt;
	int nInt2;
	int k;
	int b_y;
	coder::array<creal_T, 1U> fv;
	coder::array<creal_T, 1U> b_fv;
	coder::array<creal_T, 1U> r;
	int xoff;
	int minNrowsNx;
	int b_k;
	int b_idx;
	double im;
	double re;
	nInt2m1 = (nfft + nfft) - 1;
	wwc.set_size(nInt2m1);
	idx = nfft;
	rt = 0;
	wwc[nfft - 1].re = 1.0;
	wwc[nfft - 1].im = 0.0;
	nInt2 = nfft << 1;
	for (k = 0; k <= nfft - 2; k++) {
		double nt_im;
		double nt_re;
		b_y = ((k + 1) << 1) - 1;
		if (nInt2 - rt <= b_y) {
			rt += b_y - nInt2;
		}
		else {
			rt += b_y;
		}

		nt_im = 3.1415926535897931 * static_cast<double>(rt) / static_cast<double>
			(nfft);
		if (nt_im == 0.0) {
			nt_re = 1.0;
			nt_im = 0.0;
		}
		else {
			nt_re = std::cos(nt_im);
			nt_im = std::sin(nt_im);
		}

		wwc[idx - 2].re = nt_re;
		wwc[idx - 2].im = -nt_im;
		idx--;
	}

	idx = 0;
	b_y = nInt2m1 - 1;
	for (k = b_y; k >= nfft; k--) {
		wwc[k] = wwc[idx];
		idx++;
	}

	idx = x.size(0);
	y.set_size(nfft, x.size(1));
	if (nfft > x.size(0)) {
		nInt2m1 = x.size(1);
		for (b_y = 0; b_y < nInt2m1; b_y++) {
			rt = y.size(0);
			for (nInt2 = 0; nInt2 < rt; nInt2++) {
				y[nInt2 + y.size(0) * b_y].re = 0.0;
				y[nInt2 + y.size(0) * b_y].im = 0.0;
			}
		}
	}

	nInt2m1 = x.size(1) - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(fv,b_fv,r,xoff,minNrowsNx,b_k,b_idx,im,re)

	for (int chan = 0; chan <= nInt2m1; chan++) {
		xoff = chan * idx;
		r.set_size(nfft);
		if (nfft > x.size(0)) {
			r.set_size(nfft);
			for (minNrowsNx = 0; minNrowsNx < nfft; minNrowsNx++) {
				r[minNrowsNx].re = 0.0;
				r[minNrowsNx].im = 0.0;
			}
		}

		minNrowsNx = x.size(0);
		if (nfft < minNrowsNx) {
			minNrowsNx = nfft;
		}

		for (b_k = 0; b_k < minNrowsNx; b_k++) {
			b_idx = (nfft + b_k) - 1;
			r[b_k].re = wwc[b_idx].re * x[xoff].re + wwc[b_idx].im * x[xoff].im;
			r[b_k].im = wwc[b_idx].re * x[xoff].im - wwc[b_idx].im * x[xoff].re;
			xoff++;
		}

		minNrowsNx++;
		for (b_k = minNrowsNx; b_k <= nfft; b_k++) {
			r[b_k - 1].re = 0.0;
			r[b_k - 1].im = 0.0;
		}

		FFTImplementationCallback::r2br_r2dit_trig_impl((r), (n2blue), (costab),
			(sintab), (b_fv));
		FFTImplementationCallback::r2br_r2dit_trig_impl((wwc), (n2blue), (costab),
			(sintab), (fv));
		fv.set_size(b_fv.size(0));
		b_idx = b_fv.size(0);
		for (minNrowsNx = 0; minNrowsNx < b_idx; minNrowsNx++) {
			im = b_fv[minNrowsNx].re * fv[minNrowsNx].im + b_fv[minNrowsNx].im *
				fv[minNrowsNx].re;
			fv[minNrowsNx].re = b_fv[minNrowsNx].re * fv[minNrowsNx].re -
				b_fv[minNrowsNx].im * fv[minNrowsNx].im;
			fv[minNrowsNx].im = im;
		}

		FFTImplementationCallback::r2br_r2dit_trig_impl((fv), (n2blue), (costab),
			(sintabinv), (b_fv));
		if (b_fv.size(0) > 1) {
			im = 1.0 / static_cast<double>(b_fv.size(0));
			b_idx = b_fv.size(0);
			for (minNrowsNx = 0; minNrowsNx < b_idx; minNrowsNx++) {
				b_fv[minNrowsNx].re = im * b_fv[minNrowsNx].re;
				b_fv[minNrowsNx].im = im * b_fv[minNrowsNx].im;
			}
		}

		b_idx = 0;
		minNrowsNx = wwc.size(0);
		for (b_k = nfft; b_k <= minNrowsNx; b_k++) {
			r[b_idx].re = wwc[b_k - 1].re * b_fv[b_k - 1].re + wwc[b_k - 1].im *
				b_fv[b_k - 1].im;
			r[b_idx].im = wwc[b_k - 1].re * b_fv[b_k - 1].im - wwc[b_k - 1].im *
				b_fv[b_k - 1].re;
			if (r[b_idx].im == 0.0) {
				re = r[b_idx].re / static_cast<double>(nfft);
				im = 0.0;
			}
			else if (r[b_idx].re == 0.0) {
				re = 0.0;
				im = r[b_idx].im / static_cast<double>(nfft);
			}
			else {
				re = r[b_idx].re / static_cast<double>(nfft);
				im = r[b_idx].im / static_cast<double>(nfft);
			}

			r[b_idx].re = re;
			r[b_idx].im = im;
			b_idx++;
		}

		b_idx = r.size(0);
		for (minNrowsNx = 0; minNrowsNx < b_idx; minNrowsNx++) {
			y[minNrowsNx + y.size(0) * chan] = r[minNrowsNx];
		}
	}
}

void FFTImplementationCallback::generate_twiddle_tables(int nRows, boolean_T useRadix2, coder::array<double, 2U>& costab, coder::array<double, 2U>& sintab, coder::array<double, 2U>& sintabinv)
{
	double e;
	int n;
	int costab1q_size_idx_1;
	double costab1q_data[4097];
	int nd2;
	int k;
	int i;
	e = 6.2831853071795862 / static_cast<double>(nRows);
	n = nRows / 2 / 2;
	costab1q_size_idx_1 = n + 1;
	costab1q_data[0] = 1.0;
	nd2 = n / 2 - 1;
	for (k = 0; k <= nd2; k++) {
		costab1q_data[k + 1] = std::cos(e * (static_cast<double>(k) + 1.0));
	}

	nd2 += 2;
	i = n - 1;
	for (k = nd2; k <= i; k++) {
		costab1q_data[k] = std::sin(e * static_cast<double>(n - k));
	}

	costab1q_data[n] = 0.0;
	if (!useRadix2) {
		n = costab1q_size_idx_1 - 1;
		nd2 = (costab1q_size_idx_1 - 1) << 1;
		costab.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		sintab.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		costab[0] = 1.0;
		sintab[0] = 0.0;
		sintabinv.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		for (k = 0; k < n; k++) {
			sintabinv[k + 1] = costab1q_data[(n - k) - 1];
		}

		for (k = costab1q_size_idx_1; k <= nd2; k++) {
			sintabinv[k] = costab1q_data[k - n];
		}

		for (k = 0; k < n; k++) {
			costab[k + 1] = costab1q_data[k + 1];
			sintab[k + 1] = -costab1q_data[(n - k) - 1];
		}

		for (k = costab1q_size_idx_1; k <= nd2; k++) {
			costab[k] = -costab1q_data[nd2 - k];
			sintab[k] = -costab1q_data[k - n];
		}
	}
	else {
		n = costab1q_size_idx_1 - 1;
		nd2 = (costab1q_size_idx_1 - 1) << 1;
		costab.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		sintab.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		costab[0] = 1.0;
		sintab[0] = 0.0;
		for (k = 0; k < n; k++) {
			costab[k + 1] = costab1q_data[k + 1];
			sintab[k + 1] = -costab1q_data[(n - k) - 1];
		}

		for (k = costab1q_size_idx_1; k <= nd2; k++) {
			costab[k] = -costab1q_data[nd2 - k];
			sintab[k] = -costab1q_data[k - n];
		}

		sintabinv.set_size(1, 0);
	}
}

void FFTImplementationCallback::b_generate_twiddle_tables(int nRows, boolean_T useRadix2, coder::array<double, 2U>& costab, coder::array<double, 2U>& sintab, coder::array<double, 2U>& sintabinv)
{
	double e;
	int n;
	int costab1q_size_idx_1;
	double costab1q_data[4097];
	int nd2;
	int k;
	int i;
	e = 6.2831853071795862 / static_cast<double>(nRows);
	n = nRows / 2 / 2;
	costab1q_size_idx_1 = n + 1;
	costab1q_data[0] = 1.0;
	nd2 = n / 2 - 1;
	for (k = 0; k <= nd2; k++) {
		costab1q_data[k + 1] = std::cos(e * (static_cast<double>(k) + 1.0));
	}

	nd2 += 2;
	i = n - 1;
	for (k = nd2; k <= i; k++) {
		costab1q_data[k] = std::sin(e * static_cast<double>(n - k));
	}

	costab1q_data[n] = 0.0;
	if (!useRadix2) {
		n = costab1q_size_idx_1 - 1;
		nd2 = (costab1q_size_idx_1 - 1) << 1;
		costab.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		sintab.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		costab[0] = 1.0;
		sintab[0] = 0.0;
		sintabinv.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		for (k = 0; k < n; k++) {
			sintabinv[k + 1] = costab1q_data[(n - k) - 1];
		}

		for (k = costab1q_size_idx_1; k <= nd2; k++) {
			sintabinv[k] = costab1q_data[k - n];
		}

		for (k = 0; k < n; k++) {
			costab[k + 1] = costab1q_data[k + 1];
			sintab[k + 1] = -costab1q_data[(n - k) - 1];
		}

		for (k = costab1q_size_idx_1; k <= nd2; k++) {
			costab[k] = -costab1q_data[nd2 - k];
			sintab[k] = -costab1q_data[k - n];
		}
	}
	else {
		n = costab1q_size_idx_1 - 1;
		nd2 = (costab1q_size_idx_1 - 1) << 1;
		costab.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		sintab.set_size(1, (static_cast<int>(static_cast<short>(nd2 + 1))));
		costab[0] = 1.0;
		sintab[0] = 0.0;
		for (k = 0; k < n; k++) {
			costab[k + 1] = costab1q_data[k + 1];
			sintab[k + 1] = costab1q_data[(n - k) - 1];
		}

		for (k = costab1q_size_idx_1; k <= nd2; k++) {
			costab[k] = -costab1q_data[nd2 - k];
			sintab[k] = costab1q_data[k - n];
		}

		sintabinv.set_size(1, 0);
	}
}

void FFTImplementationCallback::b_r2br_r2dit_trig(const coder::array<creal_T, 2U>
	&x, int n1_unsigned, const coder::array<double, 2U> &costab, const coder::
	array<double, 2U> &sintab, coder::array<creal_T, 2U> &y)
{
	int nrows;
	int loop_ub;
	int i;
	coder::array<creal_T, 1U> r;
	int xoff;
	int i2;
	nrows = x.size(0);
	y.set_size(n1_unsigned, x.size(1));
	if (n1_unsigned > x.size(0)) {
		loop_ub = x.size(1);
		for (i = 0; i < loop_ub; i++) {
			int b_loop_ub;
			b_loop_ub = y.size(0);
			for (int i1 = 0; i1 < b_loop_ub; i1++) {
				y[i1 + y.size(0) * i].re = 0.0;
				y[i1 + y.size(0) * i].im = 0.0;
			}
		}
	}

	loop_ub = x.size(1) - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(r,xoff,i2)

	for (int chan = 0; chan <= loop_ub; chan++) {
		xoff = chan * nrows;
		FFTImplementationCallback::r2br_r2dit_trig_impl((x), (xoff), (n1_unsigned),
			(costab), (sintab), (r));
		xoff = r.size(0);
		for (i2 = 0; i2 < xoff; i2++) {
			y[i2 + y.size(0) * chan] = r[i2];
		}
	}

	if (y.size(0) > 1) {
		double b;
		b = 1.0 / static_cast<double>(y.size(0));
		loop_ub = y.size(0) * y.size(1);
		for (i = 0; i < loop_ub; i++) {
			y[i].re = b * y[i].re;
			y[i].im = b * y[i].im;
		}
	}
}

void FFTImplementationCallback::dobluesteinfft(const coder::array<creal_T, 2U>
	&x, int n2blue, int nfft, const coder::array<double, 2U> &costab, const coder::
	array<double, 2U> &sintab, const coder::array<double, 2U> &sintabinv, coder::
	array<creal_T, 2U> &y)
{
	int nInt2m1;
	coder::array<creal_T, 1U> wwc;
	int idx;
	int rt;
	int nInt2;
	int k;
	int b_y;
	coder::array<creal_T, 1U> fv;
	coder::array<creal_T, 1U> b_fv;
	coder::array<creal_T, 1U> r;
	int xoff;
	int minNrowsNx;
	int b_k;
	int b_idx;
	double im;
	nInt2m1 = (nfft + nfft) - 1;
	wwc.set_size(nInt2m1);
	idx = nfft;
	rt = 0;
	wwc[nfft - 1].re = 1.0;
	wwc[nfft - 1].im = 0.0;
	nInt2 = nfft << 1;
	for (k = 0; k <= nfft - 2; k++) {
		double nt_im;
		double nt_re;
		b_y = ((k + 1) << 1) - 1;
		if (nInt2 - rt <= b_y) {
			rt += b_y - nInt2;
		}
		else {
			rt += b_y;
		}

		nt_im = -3.1415926535897931 * static_cast<double>(rt) / static_cast<double>
			(nfft);
		if (nt_im == 0.0) {
			nt_re = 1.0;
			nt_im = 0.0;
		}
		else {
			nt_re = std::cos(nt_im);
			nt_im = std::sin(nt_im);
		}

		wwc[idx - 2].re = nt_re;
		wwc[idx - 2].im = -nt_im;
		idx--;
	}

	idx = 0;
	b_y = nInt2m1 - 1;
	for (k = b_y; k >= nfft; k--) {
		wwc[k] = wwc[idx];
		idx++;
	}

	idx = x.size(0);
	y.set_size(nfft, x.size(1));
	if (nfft > x.size(0)) {
		nInt2m1 = x.size(1);
		for (b_y = 0; b_y < nInt2m1; b_y++) {
			rt = y.size(0);
			for (nInt2 = 0; nInt2 < rt; nInt2++) {
				y[nInt2 + y.size(0) * b_y].re = 0.0;
				y[nInt2 + y.size(0) * b_y].im = 0.0;
			}
		}
	}

	nInt2m1 = x.size(1) - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(fv,b_fv,r,xoff,minNrowsNx,b_k,b_idx,im)

	for (int chan = 0; chan <= nInt2m1; chan++) {
		xoff = chan * idx;
		r.set_size(nfft);
		if (nfft > x.size(0)) {
			r.set_size(nfft);
			for (minNrowsNx = 0; minNrowsNx < nfft; minNrowsNx++) {
				r[minNrowsNx].re = 0.0;
				r[minNrowsNx].im = 0.0;
			}
		}

		minNrowsNx = x.size(0);
		if (nfft < minNrowsNx) {
			minNrowsNx = nfft;
		}

		for (b_k = 0; b_k < minNrowsNx; b_k++) {
			b_idx = (nfft + b_k) - 1;
			r[b_k].re = wwc[b_idx].re * x[xoff].re + wwc[b_idx].im * x[xoff].im;
			r[b_k].im = wwc[b_idx].re * x[xoff].im - wwc[b_idx].im * x[xoff].re;
			xoff++;
		}

		minNrowsNx++;
		for (b_k = minNrowsNx; b_k <= nfft; b_k++) {
			r[b_k - 1].re = 0.0;
			r[b_k - 1].im = 0.0;
		}

		FFTImplementationCallback::r2br_r2dit_trig_impl((r), (n2blue), (costab),
			(sintab), (b_fv));
		FFTImplementationCallback::r2br_r2dit_trig_impl((wwc), (n2blue), (costab),
			(sintab), (fv));
		fv.set_size(b_fv.size(0));
		b_idx = b_fv.size(0);
		for (minNrowsNx = 0; minNrowsNx < b_idx; minNrowsNx++) {
			im = b_fv[minNrowsNx].re * fv[minNrowsNx].im + b_fv[minNrowsNx].im *
				fv[minNrowsNx].re;
			fv[minNrowsNx].re = b_fv[minNrowsNx].re * fv[minNrowsNx].re -
				b_fv[minNrowsNx].im * fv[minNrowsNx].im;
			fv[minNrowsNx].im = im;
		}

		FFTImplementationCallback::r2br_r2dit_trig_impl((fv), (n2blue), (costab),
			(sintabinv), (b_fv));
		if (b_fv.size(0) > 1) {
			im = 1.0 / static_cast<double>(b_fv.size(0));
			b_idx = b_fv.size(0);
			for (minNrowsNx = 0; minNrowsNx < b_idx; minNrowsNx++) {
				b_fv[minNrowsNx].re = im * b_fv[minNrowsNx].re;
				b_fv[minNrowsNx].im = im * b_fv[minNrowsNx].im;
			}
		}

		b_idx = 0;
		minNrowsNx = wwc.size(0);
		for (b_k = nfft; b_k <= minNrowsNx; b_k++) {
			r[b_idx].re = wwc[b_k - 1].re * b_fv[b_k - 1].re + wwc[b_k - 1].im *
				b_fv[b_k - 1].im;
			r[b_idx].im = wwc[b_k - 1].re * b_fv[b_k - 1].im - wwc[b_k - 1].im *
				b_fv[b_k - 1].re;
			b_idx++;
		}

		b_idx = r.size(0);
		for (minNrowsNx = 0; minNrowsNx < b_idx; minNrowsNx++) {
			y[minNrowsNx + y.size(0) * chan] = r[minNrowsNx];
		}
	}
}

void FFTImplementationCallback::get_algo_sizes(int nfft, boolean_T useRadix2,
	int *n2blue, int *nRows)
{
	*n2blue = 1;
	if (useRadix2) {
		*nRows = nfft;
	}
	else {
		if (nfft > 0) {
			int n;
			int pmax;
			n = (nfft + nfft) - 1;
			pmax = 31;
			if (n <= 1) {
				pmax = 0;
			}
			else {
				int pmin;
				boolean_T exitg1;
				pmin = 0;
				exitg1 = false;
				while ((!exitg1) && (pmax - pmin > 1)) {
					int k;
					int pow2p;
					k = (pmin + pmax) >> 1;
					pow2p = 1 << k;
					if (pow2p == n) {
						pmax = k;
						exitg1 = true;
					}
					else if (pow2p > n) {
						pmax = k;
					}
					else {
						pmin = k;
					}
				}
			}

			*n2blue = 1 << pmax;
		}

		*nRows = *n2blue;
	}
}

void FFTImplementationCallback::r2br_r2dit_trig(const coder::array<creal_T, 2U>
	&x, int n1_unsigned, const coder::array<double, 2U> &costab, const coder::
	array<double, 2U> &sintab, coder::array<creal_T, 2U> &y)
{
	int nrows;
	int loop_ub;
	coder::array<creal_T, 1U> r;
	int xoff;
	int i2;
	nrows = x.size(0);
	y.set_size(n1_unsigned, x.size(1));
	if (n1_unsigned > x.size(0)) {
		loop_ub = x.size(1);
		for (int i = 0; i < loop_ub; i++) {
			int b_loop_ub;
			b_loop_ub = y.size(0);
			for (int i1 = 0; i1 < b_loop_ub; i1++) {
				y[i1 + y.size(0) * i].re = 0.0;
				y[i1 + y.size(0) * i].im = 0.0;
			}
		}
	}

	loop_ub = x.size(1) - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(r,xoff,i2)

	for (int chan = 0; chan <= loop_ub; chan++) {
		xoff = chan * nrows;
		FFTImplementationCallback::r2br_r2dit_trig_impl((x), (xoff), (n1_unsigned),
			(costab), (sintab), (r));
		xoff = r.size(0);
		for (i2 = 0; i2 < xoff; i2++) {
			y[i2 + y.size(0) * chan] = r[i2];
		}
	}
}

// End of code generation (FFTImplementationCallback.cpp)
