/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install, copy or use the software.
//
//
//                           License Agreement
//                For Open Source Digital Holographic Library
//
// Openholo library is free software;
// you can redistribute it and/or modify it under the terms of the BSD 2-Clause license.
//
// Copyright (C) 2017-2024, Korea Electronics Technology Institute. All rights reserved.
// E-mail : contact.openholo@gmail.com
// Web : http://www.openholo.org
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  1. Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holder or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// This software contains opensource software released under GNU Generic Public License,
// NVDIA Software License Agreement, or CUDA supplement to Software License Agreement.
// Check whether software you use contains licensed software.
//
//M*/

#ifndef __struct_h
#define __struct_h

#pragma pack(push,1)
typedef struct fileheader {
	uint8_t signature[2];
	uint32_t filesize;
	uint32_t reserved;
	uint32_t fileoffset_to_pixelarray;
} _fileheader;
typedef struct bitmapinfoheader {
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
} _bitmapinfoheader;
typedef struct rgbquad {
	uint8_t rgbBlue;
	uint8_t rgbGreen;
	uint8_t rgbRed;
	uint8_t rgbReserved;
} _rgbquad;
typedef struct bitmap {
	fileheader _fileheader;
	bitmapinfoheader _bitmapinfoheader;
} _bitmap;
typedef struct bitmap8bit {
	fileheader _fileheader;
	bitmapinfoheader _bitmapinfoheader;
	rgbquad _rgbquad[256];
} _bitmap8bit;


typedef struct Point {
	Real pos[3];
} _Point;

typedef struct Pointf {
	float pos[3];
} _Pointf;

typedef struct Color {
	Real color[3];
} _Color;

typedef struct Colorf {
	float color[3];
} _Colorf;

typedef struct Vertex {
	Point point;
	Color color;
	Real phase;
} _Vertex;

typedef struct Vertexf {
	Pointf point;
	Colorf color;
	float phase;
} _Vertexf;


typedef struct Face {
	ulonglong idx;
	Vertex vertices[3];
} _Face;

typedef struct Facef {
	ulonglong idx;
	Vertexf vertices[3];
} _Facef;

#pragma pack(pop)

#endif // !__struct_h