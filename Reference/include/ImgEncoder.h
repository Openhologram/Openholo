#pragma once
#include <atlbase.h>
class ImgEncoder
{
private:
	ImgEncoder();
	~ImgEncoder();
	static ImgEncoder *instance;
	static void Destroy() {
		delete instance;
	}
public:
	static ImgEncoder* getInstance() {
		if (instance == nullptr) {
			instance = new ImgEncoder();
			atexit(Destroy);
		}
		return instance;
	}
	bool SaveJPG(const char *path, BYTE *pBuf, UINT len);
	bool SaveGIF(const char *path, BYTE *pBuf, UINT len);
	bool SavePNG(const char *path, BYTE *pBuf, UINT len);

private:
	int GetEncoderClsid(const WCHAR *format, CLSID *pClsid);

};

