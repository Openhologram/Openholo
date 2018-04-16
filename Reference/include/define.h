#ifndef __define_h
#define __define_h

namespace oph
{
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif


//Convert Angle double
#define RADIAN(theta) (theta*M_PI)/180.0
#define DEGREE(theta) (theta*180.0)/M_PI

//-				float
#define RADIAN_F(theta) (theta*M_PI_F)/180.f
#define DEGREE_F(theta) (theta*180.f)/M_PI_F

//Mode Flag
#define MODE_CPU 1
#define MODE_GPU 0
}

#endif // !__define_h