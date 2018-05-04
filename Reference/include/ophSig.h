/**
* @mainpage ophSig
* @brief Abstract class for core processing classes
*/

#ifndef __ophSig_h
#define __ophSig_h

#include "Openholo.h"

#ifdef SIG_EXPORT
#define SIG_DLL __declspec(dllexport)
#else
#define SIG_DLL __declspec(dllimport)
#endif

class SIG_DLL ophSig : public Openholo
{
public:
	/**
	* @brief Constructor
	*/
	ophSig(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophSig(void);

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void) = 0;
};

#endif // !__ophSig_h
