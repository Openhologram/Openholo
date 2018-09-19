#ifndef __ophDepthMap_h
#define __ophDepthMap_h

#include "ophGen.h"
#include <cufft.h>

#include "include.h"

using namespace oph;

class GEN_DLL ophDepthMap : public ophGen {

public:
	explicit ophDepthMap();

protected:
	virtual ~ophDepthMap();

public:

	/** \ingroup init_module */
	void setMode(bool is_CPU);
	bool readConfig(const char* fname);
	bool readImageDepth(const char* source_folder, const char* img_prefix, const char* depth_img_prefix);

	/** \ingroup gen_module */
	Real generateHologram(void);

	/** \ingroup encode_module */
	void encodeHologram(void);

	/** \ingroup write_module */
	virtual int save(const char* fname, uint8_t bitsperpixel = 24);

public:
	/** \ingroup getter/setter */
	inline void setFieldLens(Real fieldlens) { dm_config_.field_lens = fieldlens; }
	/** \ingroup getter/setter */
	inline void setNearDepth(Real neardepth) { dm_config_.near_depthmap = neardepth; }
	/** \ingroup getter/setter */
	inline void setFarDepth(Real fardetph) { dm_config_.far_depthmap = fardetph; }
	/** \ingroup getter/setter */
	inline void setNumOfDepth(uint numofdepth) { dm_config_.num_of_depth = numofdepth; }

	/** \ingroup getter/setter */
	inline Real getFieldLens(void) { return dm_config_.field_lens; }
	/** \ingroup getter/setter */
	inline Real getNearDepth(void) { return dm_config_.near_depthmap; }
	/** \ingroup getter/setter */
	inline Real getFarDepth(void) { return dm_config_.far_depthmap; }
	/** \ingroup getter/setter */
	inline uint getNumOfDepth(void) { return dm_config_.num_of_depth; }
	/** \ingroup getter/setter */
	inline void getRenderDepth(std::vector<int>& renderdepth) { renderdepth = dm_config_.render_depth; }
	
private:

	/** \ingroup init_module
	* @{ */
	void initialize();
	void initCPU();   
	void initGPU();
	/** @} */

	/** \ingroup load_module
	* @{ */
	bool prepareInputdataCPU(uchar* img, uchar* dimg);
	bool prepareInputdataGPU(uchar* img, uchar* dimg);
	/** @} */

	/** \ingroup depth_module
	* @{ */
	void getDepthValues();
	void changeDepthQuanCPU();
	void changeDepthQuanGPU();
	/** @} */

	/** \ingroup trans_module
	* @{ */
	void transformViewingWindow();
	/** @} */

	/** \ingroup gen_module 
	* @{ */
	void calcHoloByDepth(void);
	void calcHoloCPU(void);
	void calcHoloGPU(void);
	void propagationAngularSpectrumGPU(cufftDoubleComplex* input_u, Real propagation_dist);

protected:
	void free_gpu(void);

	void ophFree(void);

private:
	bool					is_CPU;								///< if true, it is implemented on the CPU, otherwise on the GPU.

	unsigned char*			img_src_gpu;						///< GPU variable - image source data, values are from 0 to 255.
	unsigned char*			dimg_src_gpu;						///< GPU variable - depth map data, values are from 0 to 255.
	Real*					depth_index_gpu;					///< GPU variable - quantized depth map data.

	Real*					img_src;							///< CPU variable - image source data, values are from 0 to 1.
	Real*					dmap_src;							///< CPU variable - depth map data, values are from 0 to 1.
	Real*					depth_index;						///< CPU variable - quantized depth map data.
	int*					alpha_map;							///< CPU variable - calculated alpha map data, values are 0 or 1.

	Real*					dmap;								///< CPU variable - physical distances of depth map.

	Real					dstep;								///< the physical increment of each depth map layer.
	std::vector<Real>		dlevel;								///< the physical value of all depth map layer.
	std::vector<Real>		dlevel_transform;					///< transfomed dlevel variable

	OphDepthMapConfig		dm_config_;							///< structure variable for depthmap hologram configuration.
};


#endif //>__ophDepthMap_h