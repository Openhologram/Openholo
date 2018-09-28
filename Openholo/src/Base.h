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

#ifndef __Base_h
#define __Base_h

#ifdef OPH_EXPORT
#define OPH_DLL __declspec(dllexport)
#else
#define OPH_DLL __declspec(dllimport)
#endif

class OPH_DLL Base {
public:
	/**
	* @brief Constructor
	* @details Initialize variable.
	*/
	inline explicit Base(void) : refCnt(0) {}
protected:
	/**
	* @brief Destructor
	*/
	inline virtual ~Base(void) {}

protected:
	unsigned long refCnt;

public:
	/**
	* @brief If referenced this(Base's child, not abstract class) instance, must call this method.
	*/
	inline unsigned long addRef(void) { return ++refCnt; }

	/**
	* @brief Call release() when reference is finished.
	*/
	inline unsigned long release(void) {
		if (!refCnt) {
			ophFree();
			delete this;
			return 0;
		}
		return refCnt--;
	}

protected:
	/**
	* @brief When refCnt is 0 (zero), it is call inside release() when release() is called. 
	* @details A class inheriting from Base can override this method.
	*/
	virtual void ophFree(void) = 0 {}
};

#endif // !__Base_h