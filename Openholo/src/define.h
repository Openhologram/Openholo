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

#define OPH_FORWARD (-1)
#define OPH_BACKWARD (1)

#define OPH_MEASURE (0U)
#define OPH_DESTROY_INPUT (1U << 0)
#define OPH_UNALIGNED (1U << 1)
#define OPH_CONSERVE_MEMORY (1U << 2)
#define OPH_EXHAUSTIVE (1U << 3)
#define OPH_PRESERVE_INPUT (1U << 4)
#define OPH_PATIENT (1U << 5)
#define OPH_ESTIMATE (1U << 6)
#define OPH_WISDOM_ONLY (1U << 21)

#ifndef _X
#define _X 0
#endif

#ifndef _Y
#define _Y 1
#endif

#ifndef _Z
#define _Z 2
#endif

#ifndef _W
#define _W 3
#endif

#ifndef MAX_FLOAT
#define MAX_FLOAT	((float)3.40282347e+38)
#endif

#ifndef MAX_DOUBLE
#define MAX_DOUBLE	((double)1.7976931348623158e+308)
#endif

#ifndef MIN_FLOAT
#define MIN_FLOAT	((float)1.17549435e-38)
#endif

#ifndef MIN_DOUBLE
#define MIN_DOUBLE	((double)2.2250738585072014e-308)
#endif

#define MIN_REAL MIN_DOUBLE;
#define MAX_REAL MAX_DOUBLE;


//Mode Flag
#define MODE_CPU 1
#define MODE_GPU 0

#define WIDTHBYTES(bits) (((bits)+31)/32*4)

#define OPH_PLANES 1
#define OPH_COMPRESSION 0
#define X_PIXEL_PER_METER 0x130B //2835 , 72 DPI
#define Y_PIXEL_PER_METER 0x130B //2835 , 72 DPI
}

#endif // !__define_h