#ifndef __struct_h
#define __struct_h

#pragma pack(push,1)
typedef struct {
	uint8_t signature[2];
	uint32_t filesize;
	uint32_t reserved;
	uint32_t fileoffset_to_pixelarray;
} fileheader;
typedef struct {
	uint32_t dibheadersize;
	uint32_t width;
	uint32_t height;
	uint16_t planes;
	uint16_t bitsperpixel;
	uint32_t compression;
	uint32_t imagesize;
	uint32_t ypixelpermeter;
	uint32_t xpixelpermeter;
	uint32_t numcolorspallette;
	uint32_t mostimpcolor;
} bitmapinfoheader;
typedef struct {
	uint8_t rgbBlue;
	uint8_t rgbGreen;
	uint8_t rgbRed;
	uint8_t rgbReserved;
} rgbquad;
typedef struct {
	fileheader fileheader;
	bitmapinfoheader bitmapinfoheader;
	rgbquad rgbquad[256];
} bitmap;
#pragma pack(pop)

#endif // !__struct_h