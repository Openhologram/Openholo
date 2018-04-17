/**
* @mainpage OphReconstruction
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

class RECON_DLL OphReconstruction : public Openholo
{
public:
	/**
	* @brief Constructor
	*/
	OphReconstruction(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~OphReconstruction(void);

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void) = 0;
};

#endif // !__OphReconstruction_h