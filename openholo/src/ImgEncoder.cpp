#include "ImgEncoder.h"
#include <string.h>
#include <gdiplus.h>

#pragma comment(lib, "gdiplus.lib")
using namespace Gdiplus;
ImgEncoder* ImgEncoder::instance = nullptr;

ImgEncoder::ImgEncoder()
{
}


ImgEncoder::~ImgEncoder()
{
}

bool ImgEncoder::SaveJPG(const char *path, BYTE *pSrc, UINT len)
{
	bool bOK = false;
	BYTE *pDest = new BYTE[len];
	memcpy(pDest, pSrc, len);
	
	CComPtr<IStream> pStream;
	pStream.Attach(SHCreateMemStream(pDest, len));
	Image img(pStream, FALSE);
	GdiplusStartupInput gsi;
	ULONG_PTR token = NULL;

	if (GdiplusStartup(&token, &gsi, NULL) == Ok) {
		
		CLSID clsid;
		if (GetEncoderClsid(L"image/jpeg", &clsid) != -1) {
			if (img.Save(CA2W(path), &clsid, NULL) != Ok)
				bOK = true;
		}
	}
	GdiplusShutdown(token);
	pStream.Detach();
	
	delete[] pDest;
	return bOK;
}

bool ImgEncoder::SavePNG(const char *path, BYTE *pSrc, UINT len)
{
	bool bOK = false;
	BYTE *pDest = new BYTE[len];
	memcpy(pDest, pSrc, len);

	CComPtr<IStream> pStream;
	pStream.Attach(SHCreateMemStream(pDest, len));
	Image img(pStream, FALSE);
	GdiplusStartupInput gsi;
	ULONG_PTR token = NULL;

	if (GdiplusStartup(&token, &gsi, NULL) == Ok) {
		CLSID clsid;
		if (GetEncoderClsid(L"image/png", &clsid) != -1) {
			if (img.Save(CA2W(path), &clsid, NULL) != Ok)
				bOK = true;
		}
	}
	GdiplusShutdown(token);
	pStream.Detach();

	delete[] pDest;
	return bOK;
}

bool ImgEncoder::SaveGIF(const char *path, BYTE *pSrc, UINT len)
{
	bool bOK = false;
	BYTE *pDest = new BYTE[len];
	memcpy(pDest, pSrc, len);

	CComPtr<IStream> pStream;
	pStream.Attach(SHCreateMemStream(pDest, len));
	Image img(pStream, FALSE);
	GdiplusStartupInput gsi;
	ULONG_PTR token = NULL;

	if (GdiplusStartup(&token, &gsi, NULL) == Ok) {
		CLSID clsid;
		if (GetEncoderClsid(L"image/gif", &clsid) != -1) {
			if (img.Save(CA2W(path), &clsid, NULL) != Ok)
				bOK = true;
		}
	}
	GdiplusShutdown(token);
	pStream.Detach();

	delete[] pDest;
	return bOK;
}

int ImgEncoder::GetEncoderClsid(const WCHAR *format, CLSID *pClsid)
{
	UINT  num = 0;            // number of image encoders
	UINT  size = 0;           // size of the image encoder array in bytes

	ImageCodecInfo* pImageCodecInfo = NULL;

	GetImageEncodersSize(&num, &size);
	if (size == 0)
		return -1;  // Failure

	pImageCodecInfo = (ImageCodecInfo*)malloc(size);
	if (pImageCodecInfo == NULL)
		return -1;  // Failure

	GetImageEncoders(num, size, pImageCodecInfo);

	for (UINT j = 0; j < num; ++j)
	{
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;                                         // Success
		}
	}
	free(pImageCodecInfo);
	return -1;
}
