#pragma once


#ifndef __ophSigPU_h
#define __ophSigPU_h

#include "ophSig.h"

/**
* @addtogroup PU
//@{
* @details
This module contains methods related to phase unwrapping.
Phase unwrapping algorithm unwraps the 2-pi wrapped phase data to obtain continuous phase distribution.
This module uses the Goldstein's branch cut algorithm. R.M.Goldstein, H.A. Zebker, and C.L. Werner, "Satellite radar interferometry: two-dimensional phase unwrapping," Radio Science, vol. 23, no. 4, pp. 713-720, 1988.
This module was written with slight modifications based on 
https://kr.mathworks.com/matlabcentral/fileexchange/22504-2d-phase-unwrapping-algorithms
*/
//! @} PU

/**
* @ingroup PU
* @brief
* @author
*/
class SIG_DLL ophSigPU : public ophSig
{
public:
	explicit ophSigPU(void);

	/**
	* @brief Set parameters for Goldstein branchcut algorithm 
	* @param maxBoxRadius : maximum box radius for neighboring residue search
	*/
	bool setPUparam(int maxBoxRadius);

	/**
	* @brief Load original wrapped phase data
	* @param fname : image file name of wrapped phase data
	* @param bitpixel : the number of bits per pixel in the image file
	*/
	bool loadPhaseOriginal(const char *fname, int bitpixel);
	bool loadPhaseOriginal(void);

	/**
	* @brief Run phase unwrapping algorithm
	*/
	bool runPU(void);

	/**
	* @brief Save the unwrapped phase data to image file
	* @param fname : image file name where the unwrapped phase data will be stored
	*/
	bool savePhaseUnwrapped(const char *fname);

	/**
	* @brief Read configure file
	* @param fname : configure file name
	*/
	bool readConfig(const char * fname);

protected:

	virtual ~ophSigPU(void) = default;
	void phaseResidues(matrix<Real> &outputResidue);
	void branchCuts(matrix<Real> &inputResidue, matrix<Real> &outputBranchCuts);
	void placeBranchCutsInternal(matrix<Real> &branchCuts, int r1, int c1, int r2, int c2);
	void floodFill(matrix<Real> &inputBranchCuts);
	double unwrap(double phaseRef, double phaseInput);
	double mod2pi(double phase);
	void findNZ(matrix<int> &inputMatrix, vector<int> &row, vector<int> &col);
	int matrixPartialSum(matrix<int> &inputMatrix, int r1, int c1, int r2, int c2);
public:



protected:
	int MaxBoxRadius;
	int Nr;
	int Nc;
	matrix<Real> PhaseOriginal;
	matrix<Real> PhaseUnwrapped;
};

#endif // !__ophSigPU_h
