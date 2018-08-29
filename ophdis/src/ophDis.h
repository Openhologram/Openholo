#ifndef __OphDisplay_h
#define __OphDisplay_h

#include "ophRec.h"

#ifdef DISP_EXPORT
#define DISP_DLL __declspec(dllexport)
#else
#define DISP_DLL __declspec(dllimport)
#endif

class DISP_DLL ophDis : public ophRec
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophDis(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophDis(void);

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void) = 0;
};

#endif // !__OphDisplay_h