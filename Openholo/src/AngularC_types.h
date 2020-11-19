//
//  AngularC_types.h
//
//  Code generation for function 'AngularC_types'
//


#ifndef ANGULARC_TYPES_H
#define ANGULARC_TYPES_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#ifdef _MSC_VER

#pragma warning(push)
#pragma warning(disable : 4251)

#endif

// Type Declarations
class FFTImplementationCallback;
class AngularSpectrum;

// Type Definitions
class __declspec(dllexport) FFTImplementationCallback
{
 public:
  static void get_algo_sizes(int nfft, boolean_T useRadix2, int *n2blue, int
    *nRows);
  static void r2br_r2dit_trig(const coder::array<creal_T, 2U> &x, int
    n1_unsigned, const coder::array<double, 2U> &costab, const coder::array<
    double, 2U> &sintab, coder::array<creal_T, 2U> &y);
  static void dobluesteinfft(const coder::array<creal_T, 2U> &x, int n2blue, int
    nfft, const coder::array<double, 2U> &costab, const coder::array<double, 2U>
    &sintab, const coder::array<double, 2U> &sintabinv, coder::array<creal_T, 2U>
    &y);
  static void b_r2br_r2dit_trig(const coder::array<creal_T, 2U> &x, int
    n1_unsigned, const coder::array<double, 2U> &costab, const coder::array<
    double, 2U> &sintab, coder::array<creal_T, 2U> &y);
  static void b_dobluesteinfft(const coder::array<creal_T, 2U> &x, int n2blue,
    int nfft, const coder::array<double, 2U> &costab, const coder::array<double,
    2U> &sintab, const coder::array<double, 2U> &sintabinv, coder::array<creal_T,
    2U> &y);
  static void generate_twiddle_tables(int nRows, boolean_T
	  useRadix2, coder::array<double, 2U> &costab, coder::array<double, 2U> &sintab,
	  coder::array<double, 2U> &sintabinv);
  static void b_generate_twiddle_tables(int nRows, boolean_T
	  useRadix2, coder::array<double, 2U> &costab, coder::array<double, 2U> &sintab,
	  coder::array<double, 2U> &sintabinv);
 protected:
  static void r2br_r2dit_trig_impl(const coder::array<creal_T, 2U> &x, int
    xoffInit, int unsigned_nRows, const coder::array<double, 2U> &costab, const
    coder::array<double, 2U> &sintab, coder::array<creal_T, 1U> &y);
  static void r2br_r2dit_trig_impl(const coder::array<creal_T, 1U> &x, int
    unsigned_nRows, const coder::array<double, 2U> &costab, const coder::array<
    double, 2U> &sintab, coder::array<creal_T, 1U> &y);
};

#define MAX_THREADS                    omp_get_max_threads()
#ifdef _MSC_VER

#pragma warning(pop)

#endif
#endif

// End of code generation (AngularC_types.h)
