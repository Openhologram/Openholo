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
	OphReconstruction(void);

protected:
	virtual ~OphReconstruction(void);

protected:
	virtual void ophFree(void) = 0;
};

#endif // !__OphReconstruction_h