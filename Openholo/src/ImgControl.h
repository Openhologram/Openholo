#pragma once
#ifndef __ImgControl_h
#define __ImgControl_h

#include <atlbase.h>


#ifdef OPH_EXPORT
#define OPH_DLL __declspec(dllexport)
#else
#define OPH_DLL __declspec(dllimport)
#endif
namespace oph
{
	enum FLIP
	{
		NONE,
		VERTICAL,
		HORIZONTAL,
		BOTH
	};

	enum TYPE
	{
		BMP,
		JPG,
		PNG,
		TIFF,
		GIF
	};

	class OPH_DLL ImgControl
	{
	private:
		ImgControl();
		~ImgControl();
		static ImgControl *instance;
		static void Destroy() {
			delete instance;
		}
	public:
		static ImgControl* getInstance() {
			if (instance == nullptr) {
				instance = new ImgControl();
				atexit(Destroy);
			}
			return instance;
		}

		bool Save(const char *path, BYTE *pBuf, UINT len, int quality = 100);
		int CalcBitmapSize(int w, int h, int ch) { return (((w * ch) + 3) & ~3) * h; }
		bool GetSize(const char* path, unsigned int *size);
		void Resize(unsigned char* src, unsigned char* dst, int w, int h, int neww, int newh, int ch);
		bool Rotate(double rotate, unsigned char *src, unsigned char *dst, int w, int h, int neww, int newh, int ch);
		bool Flip(FLIP mode, unsigned char *src, unsigned char *dst, int w, int h, int ch);
		bool Crop(unsigned char *src, unsigned char *dst, int w, int h, int ch, int x, int y, int neww, int newh);


	private:
		int GetEncoderClsid(const WCHAR *format, CLSID *pClsid);
		char* GetExtension(const char *path);
	};
}
#endif