#ifndef __OphCoreProcessing_h
#define __OphCoreProcessing_h

#include "Openholo.h"

#ifdef COREPROCSS_EXPORT
#define COREPROCSS_DLL __declspec(dllexport)
#else
#define COREPROCSS_DLL __declspec(dllimport)
#endif

class COREPROCSS_DLL OphCoreProcessing : public Openholo
{
public:
	OphCoreProcessing(void);

protected:
	virtual ~OphCoreProcessing(void);

protected:
	virtual void ophFree(void) = 0;
};

#endif // !__OphCoreProcessing_h
