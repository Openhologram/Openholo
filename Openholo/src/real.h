#ifndef __real_h
#define __real_h

typedef double real;
typedef float  real_t;

#define REAL_T_IS_FLOAT 1

namespace oph {

#ifndef _MAXFLOAT
#define _MAXFLOAT	((float)3.40282347e+38)
#endif

#ifndef _MAXDOUBLE
#define _MAXDOUBLE	((double)1.7976931348623158e+308)
#endif

#define _MINFLOAT	((float)1.17549435e-38)
#define _MINDOUBLE	((double)2.2250738585072014e-308)

#ifndef M_PI
#define M_PI		 3.141592653589793238462643383279502884197169399375105820974944592308
#endif

#define MINREAL _MINDOUBLE;
#define MAXREAL _MAXDOUBLE;


};

#endif // !__real_h
