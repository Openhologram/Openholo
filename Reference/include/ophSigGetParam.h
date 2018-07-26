/**
* @mainpage ophSigGetParam
* @brief The class for extracting of the depth parameter of the hologram
*/

#ifndef __ophSigGetParam_h
#define __ophSigGetParam_h

#include "ophSig.h"

class SIG_DLL ophSigGetParam : public ophSig
{
public:
	/**
	* @brief Constructor
	*/
	ophSigGetParam(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophSigGetParam(void) = default;

public:
	virtual bool loadParam(std::string cfg);

	float sigGetParamSF(float zMax, float zMin, int sampN, float th = 1);
	float sigGetParamAT();
	float sigGetParamAT(float lambda);

private:
	virtual void ophFree(void);

	float _zMax;
	float _zMin;
	int _sampN;
};

#endif // !__ophSigGetParam_h
