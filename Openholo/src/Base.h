/*!
* \file Base.h
* \date 2018/04/10
*
* \author Kim Ryeonwoo
* Contact: kimlw90@keti.re.kr
*
* \brief
*
* TODO: Top class.
		Check reference count for prevention memory leak,		
		if reference to this object, must call addRef(),
		call release() must be called after the call is finished.
*
* \note
*/


#ifndef __Base_h
#define __Base_h

#ifdef OPH_EXPORT
#define OPH_DLL __declspec(dllexport)
#else
#define OPH_DLL __declspec(dllimport)
#endif

class OPH_DLL Base {
public:
	inline Base(void) : refCnt(0) {}
protected:
	inline virtual ~Base(void) {}

protected:
	unsigned long refCnt;

public:
	inline unsigned long addRef(void) { return ++refCnt; }
	inline unsigned long release(void) {
		if (!refCnt)
		{
			ophFree();
			delete this;
			return 0;
		}
		return refCnt--;
	}

protected:
	virtual void ophFree(void) = 0 {}
};

#endif // !__Base_h