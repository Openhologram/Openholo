/**
* @mainpage Openholo 
* @brief Abstract class
*/

#ifndef __Openholo_h
#define __Openholo_h

#include "Base.h"
#include "include.h"

class OPH_DLL Openholo : public Base
{

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

protected:
	/**
	* @param input parameter. bitmap data
	* @param input parameter. image size, width
	* @param input parameter. image size, Height
	* @param input parameter. bits per pixel
	* @param output parameter. file name
	*/
	int createBitmapFile(unsigned char* pixelbuffer, int pic_width, int pic_height, uint16_t bitsperpixel, const char* file_name);

	/**
	* @param output parameter. image size, width
	* @param output parameter. image size, Height
	* @param output parameter. bytes per pixel
	* @param input parameter. file name
	*/
	int getBitmapSize(int& w, int& h, int& bytesperpixel, const char* file_name);

	/**
	* @param input parameter. image data buffer
	* @param input parameter. file name
	*/
	int loadBitmapFile(unsigned char* pixelbuffer, const char* file_name);

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
	* @param input parameter. dest image data
	* @param output parameter. source image data
	* @param input parameter. image size, width
	* @param input parameter. image size, Height
	* @param input parameter. bytes per pixel
	*/
	void convertToFormatGray8(unsigned char* img, unsigned char* imgload, int w, int h, int bytesperpixel);

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void) = 0;
};

#endif // !__Openholo_h