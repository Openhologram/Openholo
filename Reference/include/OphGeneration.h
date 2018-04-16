#ifndef __OphGeneration_h
#define __OphGeneration_h

#include "Openholo.h"

#ifdef GEN_EXPORT
#define GEN_DLL __declspec(dllexport)
#else
#define GEN_DLL __declspec(dllimport)
#endif

class GEN_DLL OphGeneration : public Openholo
{
public:
	OphGeneration(void);

protected:
	virtual ~OphGeneration(void);

protected:
	int loadPointCloudData(const std::string InputModelFile, std::vector<float> *VertexArray, std::vector<float> *AmplitudeArray, std::vector<float> *PhaseArray);
	bool readConfigFile(const std::string InputConfigFile, oph::OphConfigParams &configParams);

protected:
	virtual void ophFree(void) = 0;
};

#endif // !__OphGeneration_h
