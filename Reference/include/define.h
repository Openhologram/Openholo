#ifndef __define_h
#define __define_h

namespace oph
{
#ifndef M_PI
#define M_PI	3.141592653589793238462643383279502884197169399375105820974944592308
#endif

#ifndef M_PI_F
#define M_PI_F	3.14159265358979323846f
#endif

//Convert Angle double
#define RADIAN(theta)	(theta*M_PI)/180.0
#define DEGREE(theta)	(theta*180.0)/M_PI
//-				float
#define RADIAN_F(theta) (theta*M_PI_F)/180.f
#define DEGREE_F(theta) (theta*180.f)/M_PI_F

#ifndef _X
#define _X 0
#endif

#ifndef _Y
#define _Y 1
#endif

#ifndef _Z
#define _Z 2
#endif

#ifndef _MAXFLOAT
#define _MAXFLOAT	((float)3.40282347e+38)
#endif

#ifndef _MAXDOUBLE
#define _MAXDOUBLE	((double)1.7976931348623158e+308)
#endif

#ifndef _MINFLOAT
#define _MINFLOAT	((float)1.17549435e-38)
#endif

#ifndef _MINDOUBLE
#define _MINDOUBLE	((double)2.2250738585072014e-308)
#endif

#define MINREAL _MINDOUBLE;
#define MAXREAL _MAXDOUBLE;


//Mode Flag
#define MODE_CPU 1
#define MODE_GPU 0

#define WIDTHBYTES(bits) (((bits)+31)/32*4)

#define _planes 1
#define _compression 0
#define _xpixelpermeter 0x130B //2835 , 72 DPI
#define _ypixelpermeter 0x130B //2835 , 72 DPI
}

#endif // !__define_h