/**
* @mainpage ophSigConvert
* @brief The class for conversion processing of signal(hologram)
*/

#ifndef __ophSigConvert_h
#define __ophSigConvert_h

#include "ophSig.h"

class SIG_DLL ophSigConvert : public ophSig
{
public:
	/**
	* @brief Constructor
	*/
	ophSigConvert(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophSigConvert(void) = default;

public:
	virtual bool loadParam(std::string cfg);

	bool sigConvertOffaxis(void);
	bool sigConvertHPO(void);
	bool sigConvertCAC(void);

private:
	virtual void ophFree(void);

	float _angleX;
	float _angleY;
	float _redRate;
	float _radius;
	float _foc[3];
};

#endif // !__ophSigConvert_h
