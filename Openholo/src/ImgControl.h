#pragma once
#ifndef __ImgControl_h
#define __ImgControl_h
#include <stdlib.h>
#include <stdint.h>

#ifdef _WIN32
#ifdef OPH_EXPORT
#define OPH_DLL __declspec(dllexport)
#else
#define OPH_DLL __declspec(dllimport)
#endif
#else
#ifdef OPH_EXPORT
#define OPH_DLL __attribute__((visibility("default")))
#else
#define OPH_DLL
#endif
#endif

namespace oph
{
	enum OPH_DLL FLIP
	{
		NONE,
		VERTICAL,
		HORIZONTAL,
		BOTH
	};

	enum OPH_DLL TYPE
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

		// Get file size
		bool GetSize(const char* path, uint32_t *size);
		// Resize the bitmap
		bool Resize(const uint8_t* src, uint8_t* dst, const uint32_t w, const uint32_t h,
			const uint32_t neww, const uint32_t newh, const uint8_t ch);
		// Rotate the bitmap
		bool Rotate(const double rotate, const uint8_t* src, uint8_t* dst, const uint32_t w, const uint32_t h, 
			const uint32_t neww, const uint32_t newh, const uint8_t ch);
		// Flip the bitmap
		bool Flip(FLIP mode, const uint8_t* src, uint8_t* dst, const uint32_t w, const uint32_t h, 
			const uint8_t ch);
		// Crop the bitmap
		bool Crop(const uint8_t* src, uint8_t* dst, const uint32_t w, const uint32_t h, const uint8_t ch,
			const uint32_t start_x, const uint32_t start_y, const uint32_t end_x, const uint32_t end_y);

	private:
		// Get bitmap pixel size.
		uint64_t GetPixelSize(const uint32_t width, const uint32_t height, const uint8_t channel);
	};
}
#endif
