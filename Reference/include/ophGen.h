/**
* @mainpage ophGen
* @brief Abstract class for generation classes
*/

#ifndef __ophGen_h
#define __ophGen_h

#include "Openholo.h"

#ifdef GEN_EXPORT
#define GEN_DLL __declspec(dllexport)
#else
#define GEN_DLL __declspec(dllimport)
#endif

class GEN_DLL ophGen : public Openholo
{
public:
	/**
	* @brief Constructor
	*/
	ophGen(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophGen(void);

protected:
	/**
	* @param input parameter. point cloud data file name
	* @param output parameter. point cloud data, vertices(x0, y0, z0, x1, y1, z1, ...) container's pointer
	* @param output parameter. point cloud data, amplitudes container's pointer
	* @param output parameter. point cloud data, phases container's pointer
	* @return positive integer is points number of point cloud, return a negative number if the load fails
	*/
	int loadPointCloudData(const std::string InputModelFile, std::vector<float> *VertexArray, std::vector<float> *AmplitudeArray, std::vector<float> *PhaseArray);

	/**
	* @param input parameter. configuration data file name
	* @param output parameter. OphConfigParams struct variable can get configuration data
	*/
	bool readConfigFile(const std::string InputConfigFile, oph::ConfigParams &configParams);

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void) = 0;
};

#endif // !__ophGen_h
