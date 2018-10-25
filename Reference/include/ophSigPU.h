#pragma once


#ifndef __ophSigPU_h
#define __ophSigPU_h

#include "ophSig.h"

/**
* @addtogroup PU
//@{
* @detail

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

	bool setPUparam(int maxBoxRadius);
	bool loadPhaseOriginal(const char *fname, int bitpixel);
	bool loadPhaseOriginal(void);
	bool runPU(void);
	bool savePhaseUnwrapped(const char *fname);
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
