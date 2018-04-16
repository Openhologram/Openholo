#include "vec.h"

namespace oph {


const int vec2::n = 2;

bool vec2::unit() 
{ 
    real val = norm(*this);

    if (val < epsilon) return false; 
    (*this) = (*this)/val; 
    return true; 
}

real vec2::length() const 
{ 
    return norm(*this); 
}

int vec2::is_parallel( 
      // returns  1: this and other vectors are and parallel
      //         -1: this and other vectors are anti-parallel
      //          0: this and other vectors are not parallel
      //             or at least one of the vectors is zero
      const vec2& vv,
      real angle_tolerance // (default=ON_DEFAULT_ANGLE_TOLERANCE) radians
      ) const
{
  int rc = 0;
  const real ll = norm(*this) * norm(vv);
  if ( ll > 0.0 ) {
    const real cos_angle = (inner(*this, vv))/ll;
    const real cos_tol = cos(angle_tolerance);
    if ( cos_angle >= cos_tol )
      rc = 1;
    else if ( cos_angle <= -cos_tol )
      rc = -1;
  }
  return rc;
}

bool vec2::is_perpendicular(
      // returns true:  this and other vectors are perpendicular
      //         false: this and other vectors are not perpendicular
      //                or at least one of the vectors is zero
      const vec2& vv,
      real angle_tolerance // (default=ON_DEFAULT_ANGLE_TOLERANCE) radians
      ) const
{
  bool rc = false;
  const real ll = norm(*this)*norm(vv);
  if ( ll > 0.0 ) {
    if ( fabs(inner(*this, vv)/ll) <= sin(angle_tolerance) )
      rc = true;
  }
  return rc;
}

// set this vector to be perpendicular to another vector
bool vec2::perpendicular( // Result is not unitized. 
			    // returns false if input vector is zero
      const vec2& vv
      )
{
  v[1] = vv[0];
  v[0] = -vv[1];
  return (v[0] != 0.0 || v[1] != 0.0) ? true : false;
}

// set this vector to be perpendicular to a line defined by 2 points
bool vec2::perpendicular( 
      const vec2& p, 
      const vec2& q
      )
{
  return perpendicular(q-p);
}


void store(FILE* fp, const vec2& v)
{
    fprintf(fp, "(%lg", v[0]);
    for(int i = 1; i < 2;++i){
	fprintf(fp, " %lg", v[i]);
    }
    fprintf(fp, ")\n");
}

int scan(FILE* fp, const vec2& v)
{
    int a = fscanf(fp, " (");
    for(int i = 0; i < 2;++i){
	a += fscanf(fp, " %lg", const_cast<real*>(&v[i]));
    }
    a += fscanf(fp, " )");
    return a;
}

int apx_equal(const vec2& a, const vec2& b)
{
    int c = 1;

    for (int i = 0 ; i < 2 ;++i){
	c = c && apx_equal(a[i], b[i]);
    }

    return c;
}

int apx_equal(const vec2& a, const vec2& b, real eps)
{
    int c = 1;

    for (int i = 0 ; i < 2 ;++i){
	c = c && apx_equal(a[i], b[i], eps);
    }

    return c;
}




const int vec3::n = 3;

bool vec3::unit() 
{ 
    real val = norm(*this);

    if (val < epsilon) return false; 
    (*this) = (*this)/val; 
    return true; 
}

real vec3::length() const 
{ 
    return norm(*this); 
}

int vec3::is_parallel( 
      // returns  1: this and other vectors are and parallel
      //         -1: this and other vectors are anti-parallel
      //          0: this and other vectors are not parallel
      //             or at least one of the vectors is zero
      const vec3& vv,
      real angle_tolerance // (default=ON_DEFAULT_ANGLE_TOLERANCE) radians
      ) const
{
  int rc = 0;
  const real ll = norm(*this) * norm(vv);
  if ( ll > 0.0 ) {
    const real cos_angle = (inner(*this, vv))/ll;
    const real cos_tol = cos(angle_tolerance);
    if ( cos_angle >= cos_tol )
      rc = 1;
    else if ( cos_angle <= -cos_tol )
      rc = -1;
  }
  return rc;
}

bool vec3::is_perpendicular(
      // returns true:  this and other vectors are perpendicular
      //         false: this and other vectors are not perpendicular
      //                or at least one of the vectors is zero
      const vec3& vv,
      real angle_tolerance // (default=ON_DEFAULT_ANGLE_TOLERANCE) radians
      ) const
{
    bool rc = false;
    const real ll = norm(*this) * norm(vv);
    if ( ll > 0.0 ) {
	if ( fabs(inner(oph::unit(*this), oph::unit(vv))/ll) <= sin(angle_tolerance) )
	    rc = true;
    }
    return rc;
}


bool vec3::perpendicular( const vec3& vv )
{
  //bool rc = false;
    int i, j, k; 
    real a, b;
    k = 2;
    if ( fabs(vv[1]) > fabs(vv[0]) ) {
	if ( fabs(vv[2]) > fabs(vv[1]) ) {
	    i = 2;
	    j = 1;
	    k = 0;
	    a = vv[2];
	    b = -vv[1];
	}
	else if ( fabs(vv[2]) >= fabs(vv[0]) ){
	    i = 1;
	    j = 2;
	    k = 0;
	    a = vv[1];
	    b = -vv[2];
	}
	else {
	    // |vv[1]| > |vv[0]| > |vv[2]|
	    i = 1;
	    j = 0;
	    k = 2;
	    a = vv[1];
	    b = -vv[0];
	}
    }
    else if ( fabs(vv[2]) > fabs(vv[0]) ) {
	// |vv[2]| > |vv[0]| >= |vv[1]|
	i = 2;
	j = 0;
	k = 1;
	a = vv[2];
	b = -vv[0];
    }
    else if ( fabs(vv[2]) > fabs(vv[1]) ) {
	// |vv[0]| >= |vv[2]| > |vv[1]|
	i = 0;
	j = 2;
	k = 1;
	a = vv[0];
	b = -vv[2];
    }
    else {
	// |vv[0]| >= |vv[1]| >= |vv[2]|
	i = 0;
	j = 1;
	k = 2;
	a = vv[0];
	b = -vv[1];
    }

    v[i] = b;
    v[j] = a;
    v[k] = 0.0;
    return (a != 0.0) ? true : false;
}

bool
vec3::perpendicular( 
      const vec3& P0, const vec3& P1, const vec3& P2
      )
{
    // Find a the unit normal to a triangle defined by 3 points
    vec3 V0, V1, V2, N0, N1, N2;

    v[0] = v[1] = v[2] = 0.0;

    V0 = P2 - P1;
    V1 = P0 - P2;
    V2 = P1 - P0;

    N0 = cross( V1, V2 );

    if (!N0.unit())
	return false;

    N1 = cross( V2, V0 );

    if (!N1.unit())
	return false;

    N2 = cross( V0, V1 );

    if (!N2.unit())
	return false;

    const real s0 = 1.0/V0.length();
    const real s1 = 1.0/V1.length();
    const real s2 = 1.0/V2.length();

    // choose normal with smallest total error
    const real e0 = s0*fabs(inner(N0,V0)) + s1*fabs(inner(N0,V1)) + s2*fabs(inner(N0,V2));
    const real e1 = s0*fabs(inner(N1,V0)) + s1*fabs(inner(N1,V1)) + s2*fabs(inner(N1,V2));
    const real e2 = s0*fabs(inner(N2,V0)) + s1*fabs(inner(N2,V1)) + s2*fabs(inner(N2,V2));

    if ( e0 <= e1 ) {
	if ( e0 <= e2 ) {
	  *this = N0;
	}
	else {
	  *this = N2;
	}
    }
    else if (e1 <= e2) {
	*this = N1;
    }
    else {
	*this = N2;
    }

    return true;
}
 
void store(FILE* fp, const vec3& v)
{
    fprintf(fp, "(%lg", v[0]);
    for(int i = 1; i < 3;++i){
	fprintf(fp, " %lg", v[i]);
    }
    fprintf(fp, ")\n");
}

int scan(FILE* fp, const vec3& v)
{
    int a = fscanf(fp, " (");
    for(int i = 0; i < 3;++i){
	a += fscanf(fp, " %lg", const_cast<real*>(&v[i]));
    }
    a += fscanf(fp, " )");
    return a;
}

int apx_equal(const vec3& a, const vec3& b)
{
    int c = 1;

    for (int i = 0 ; i < 3 ;++i){
	c = c && apx_equal(a[i], b[i]);
    }

    return c;
}

int apx_equal(const vec3& a, const vec3& b, real eps)
{
    int c = 1;

    for (int i = 0 ; i < 3 ;++i){
	c = c && apx_equal(a[i], b[i], eps);
    }

    return c;
}



const int vec4::n = 4;

bool vec4::unit() 
{ 
    real val = norm(*this);

    if (val < epsilon) return false; 
    (*this) = (*this)/val; 
    return true; 
}

real vec4::length() const 
{ 
    return norm(*this); 
}

void store(FILE* fp, const vec4& v)
{
    fprintf(fp, "(%lg", v[0]);
    for(int i = 1; i < 4;++i){
	fprintf(fp, " %lg", v[i]);
    }
    fprintf(fp, ")\n");
}

int scan(FILE* fp, const vec4& v)
{
    int a = fscanf(fp, " (");
    for(int i = 0; i < 4;++i){
	a += fscanf(fp, " %lg", const_cast<real*>(&v[i]));
    }
    a += fscanf(fp, " )");
    return a;
}

int apx_equal(const vec4& a, const vec4& b)
{
    int c = 1;

    for (int i = 0 ; i < 4 ;++i){
	c = c && oph::apx_equal(a[i], b[i]);
    }

    return c;
}

int apx_equal(const vec4& a, const vec4& b, real eps)
{
    int c = 1;

    for (int i = 0 ; i < 4 ;++i){
	c = c && oph::apx_equal(a[i], b[i], eps);
    }

    return c;
}

vec3 cross(const vec3& a, const vec3& b)
{
    vec3 c;
    
    c(0) = a(0 + 1) * b(0 + 2) - a(0 + 2) * b(0 + 1);
    
    c(1) = a(1 + 1) * b(1 + 2) - a(1 + 2) * b(1 + 1);
    
    c(2) = a(2 + 1) * b(2 + 2) - a(2 + 2) * b(2 + 1);
    

    return c;
}

}; // namespace graphics