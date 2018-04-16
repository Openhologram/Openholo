/*!
* \file Openholo.h
* \date 2018/04/10
*
* \author Kim Ryeonwoo
* Contact: kimlw90@keti.re.kr
*
* \brief
*
* TODO: long description
*
* \note
*/


#ifndef __Openholo_h
#define __Openholo_h

#include "Base.h"
#include "include.h"

class OPH_DLL Openholo : public Base
{

public:
	Openholo(void);
protected:
	virtual ~Openholo(void);

protected:
	int createBitmapFile(unsigned char* pixelbuffer, int pic_width, int pic_height, uint16_t bitsperpixel, const char* file_name);
	int getBitmapSize(int& w, int& h, int& bytesperpixel, const char* file_name);
	int loadBitmapFile(unsigned char* pixelbuffer, const char* file_name);
	void imgScaleBilnear(unsigned char* src, unsigned char* dst, int w, int h, int neww, int newh);
	void convertToFormatGray8(unsigned char* img, unsigned char* imgload, int w, int h, int bytesperpixel);

protected:
	virtual void ophFree(void) = 0;
};

#endif // !__Openholo_h