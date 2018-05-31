/**
* @mainpage Base
* @brief Top class
* @details Class for prevention memory leak
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