/**
* @mainpage ophRec
* @brief Abstract class for reconstruction classes
*/

#ifndef __OphReconstruction_h
#define __OphReconstruction_h

#include "Openholo.h"

#ifdef RECON_EXPORT
#define RECON_DLL __declspec(dllexport)
#else
#define RECON_DLL __declspec(dllimport)
#endif

class RECON_DLL ophRec : public Openholo
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophRec(void);
	/**
	* @brief Destructor
	*/
	virtual ~ophRec(void);

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void) = 0;
};

#endif // !__OphReconstruction_h