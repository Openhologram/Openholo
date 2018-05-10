/**
* @mainpage Openholo 
* @brief Abstract class
*/

#ifndef __Openholo_h
#define __Openholo_h

#include "Base.h"
#include "include.h"
#include "vec.h"
#include "ivec.h"
#include "real.h"

struct OPH_DLL OphContext {	
	oph::ivec2		pixel_number;				///< SLM_PIXEL_NUMBER_X & SLM_PIXEL_NUMBER_Y
	oph::vec2		pixel_pitch;				///< SLM_PIXEL_PITCH_X & SLM_PIXEL_PITCH_Y

	real			k;							///< 2 * PI / lambda(wavelength)
	oph::vec2		ss;							///< pn * pp

	real			lambda;						///< wave length
};

class OPH_DLL Openholo : public Base{

public:
	/**
	* @brief Constructor
	*/
	Openholo(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~Openholo(void);

private:
	void initialize(void);

public:
	inline void setPixelNumber(uint32_t nx, uint32_t ny) { context_.pixel_number.v[0] = nx; context_.pixel_number.v[1] = ny; }
	inline void setPixelPitch(real px, real py) { context_.pixel_pitch.v[0] = px; context_.pixel_pitch.v[1] = py; }
	inline void setWaveLength(real w) { context_.lambda = w; }

	OphContext& getContext(void) { return context_; }
	void*		getBuffer(void) { return p_hologram; }

public:
	/**
	* @breif save data is Openholo::p_hologram if src is nullptr.
	*/
	int save(const char* fname, uint8_t bitsperpixel = 8, void* src = nullptr, int pic_width = 0, int pic_height = 0);

	/**
	* @breif load data from .oph or .bmp
	* @breif loaded data is stored in the Openholo::p_hologram if dst is nullptr.
	*/
	int load(const char* fname, void* dst = nullptr);

protected:
	int checkExtension(const char* fname, const char* ext);

protected:
	int saveAsOhf(const char* fname, uint8_t bitsperpixel, void* src, int pic_width, int pic_height);
	int saveAsImg(const char* fname, uint8_t bitsperpixel, void* src, int pic_width, int pic_height);

	/**
	*/
	int loadAsOhf(const char* fname, void* dst);
	int loadAsImg(const char* fname, void* dst);

	/**
	* @param output parameter. image size, width
	* @param output parameter. image size, Height
	* @param output parameter. bytes per pixel
	* @param input parameter. file name
	*/
	int getImgSize(int& w, int& h, int& bytesperpixel, const char* file_name);

	/**
	* @brief Function for change image size
	* @param input parameter. source image data
	* @param output parameter. dest image data
	* @param input parameter. original width
	* @param input parameter. original height
	* @param input parameter. width to replace
	* @param input parameter. height to replace
	*/
	void imgScaleBilnear(unsigned char* src, unsigned char* dst, int w, int h, int neww, int newh);

	/**
	* @brief Function for convert image format to gray8
	* @param input parameter. source image data
	* @param output parameter. dest image data
	* @param input parameter. image size, width
	* @param input parameter. image size, Height
	* @param input parameter. bytes per pixel
	*/
	void convertToFormatGray8(unsigned char* src, unsigned char* dst, int w, int h, int bytesperpixel);

protected:
	OphContext	context_;
	void*		p_hologram;

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void);
};

#endif // !__Openholo_h