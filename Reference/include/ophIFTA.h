#pragma once
#include "ophGen.h"

class GEN_DLL ophIFTA :	public ophGen
{
public:
	explicit ophIFTA();
	virtual ~ophIFTA();
	Real generateHologram();
	bool readConfig(const char* fname);
	bool readImage(const char* fname, bool bRGB);
	bool normalize();
	void encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND);
	uchar getMax(uchar *src, int width, int height);
	//Real getDistance() { return m_distance; };
	void setConfig(OphIFTAConfig config) { m_config = config; }
	OphIFTAConfig& getConfig() { return m_config; }
	uint* getProgress() { return &m_nProgress; }

private:
	vec2 ss;
	vec2 pp;
	ivec2 pn;

	uchar* imgRGB;
	uchar* imgDepth;
	uchar* imgOutput;
	Real nearDepth;
	Real farDepth;
	int nIteration;
	int nDepth;
	uint m_nProgress;

	int width;
	int height;
	int bytesperpixel;

	OphIFTAConfig m_config;
};

