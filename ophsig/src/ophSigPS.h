#pragma once


#ifndef __ophSigPS_h
#define __ophSigPS_h

#include "ophSig.h"

/**
* @addtogroup PS
//@{
* @detail
This module contains methods related to phase unwrapping.
Phase unwrapping algorithm unwraps the 2-pi wrapped phase data to obtain continuous phase distribution.
This module uses the Goldstein's branch cut algorithm. R.M.Goldstein, H.A. Zebker, and C.L. Werner, "Satellite radar interferometry: two-dimensional phase unwrapping," Radio Science, vol. 23, no. 4, pp. 713-720, 1988.
This module was written with slight modifications based on
https://kr.mathworks.com/matlabcentral/fileexchange/22504-2d-phase-unwrapping-algorithms
*/
//! @} PS

/**
* @ingroup PS
* @brief
* @author
*/

class SIG_DLL ophSigPS : public ophSig
{
public:
	explicit ophSigPS(void);
};

#endif