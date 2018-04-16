#ifndef __ivec_h
#define __ivec_h
//|
//| ivec : n-dimensional ivector
//|

#include <stdio.h>

namespace oph {

/**
* @brief structure for 2-dimensional integer vector and its arithmetic.
*/
struct __declspec(dllexport) ivec2 {
    int v[2];
    static const int n;
    
    inline ivec2() { };
    
    inline ivec2(int a)
    {
	 v[1 - 1] = a;  v[2 - 1] = a;  
    }
    
    inline ivec2(int v_1  , int v_2 )
    {
	 v[1 - 1] = v_1;  v[2 - 1] = v_2; 
    }

    inline ivec2(const ivec2& a)
    {
	 v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1]; 
    }

    inline ivec2& operator=(const ivec2& a)
    {
	 v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1]; 
	return *this;
    }


    inline int& operator[] (int i) { return v[i]; }
    inline const int&  operator[] (int i) const { return v[i]; }
    inline int& operator() (int i) { return v[i % 2]; }
    inline const int&  operator() (int i) const { return v[i % 2]; }
};




//| binary op : componentwise


inline ivec2 operator + (const ivec2& a, const ivec2& b)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = a[i] + b[i]; }
    return c;
}

inline ivec2 operator + (int a, const ivec2& b)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = a + b[i]; }
    return c;
}

inline ivec2 operator + (const ivec2& a, int b)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = a[i] + b; }
    return c;
}



inline ivec2 operator - (const ivec2& a, const ivec2& b)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = a[i] - b[i]; }
    return c;
}

inline ivec2 operator - (int a, const ivec2& b)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = a - b[i]; }
    return c;
}

inline ivec2 operator - (const ivec2& a, int b)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = a[i] - b; }
    return c;
}



inline ivec2 operator * (const ivec2& a, const ivec2& b)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = a[i] * b[i]; }
    return c;
}

inline ivec2 operator * (int a, const ivec2& b)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = a * b[i]; }
    return c;
}

inline ivec2 operator * (const ivec2& a, int b)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = a[i] * b; }
    return c;
}


//
//inline ivec2 operator / (const ivec2& a, const ivec2& b)
//{
//    ivec2 c;
//    for(int i = 0; i < 2;++i) { c[i] = a[i] / b[i]; }
//    return c;
//}
//
//inline ivec2 operator / (int a, const ivec2& b)
//{
//    ivec2 c;
//    for(int i = 0; i < 2;++i) { c[i] = a / b[i]; }
//    return c;
//}
//
//inline ivec2 operator / (const ivec2& a, int b)
//{
//    ivec2 c;
//    for(int i = 0; i < 2;++i) { c[i] = a[i] / b; }
//    return c;
//}



//| cumulative op : componentwise


inline ivec2 operator += (ivec2& a, const ivec2& b)
{
    return a = (a + b);
}

inline ivec2 operator += (ivec2& a, int b)
{
    return a = (a + b);
}



inline ivec2 operator -= (ivec2& a, const ivec2& b)
{
    return a = (a - b);
}

inline ivec2 operator -= (ivec2& a, int b)
{
    return a = (a - b);
}



inline ivec2 operator *= (ivec2& a, const ivec2& b)
{
    return a = (a * b);
}

inline ivec2 operator *= (ivec2& a, int b)
{
    return a = (a * b);
}



//inline ivec2 operator /= (ivec2& a, const ivec2& b)
//{
//    return a = (a / b);
//}
//
//inline ivec2 operator /= (ivec2& a, int b)
//{
//    return a = (a / b);
//}



//| logical op : componentwise


inline int operator == (const ivec2& a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] == b[i]; }
    return c;
}

inline int operator == (int a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a == b[i]; }
    return c;
}

inline int operator == (const ivec2& a, int b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] == b; }
    return c;
}



inline int operator < (const ivec2& a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] < b[i]; }
    return c;
}

inline int operator < (int a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a < b[i]; }
    return c;
}

inline int operator < (const ivec2& a, int b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] < b; }
    return c;
}



inline int operator <= (const ivec2& a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] <= b[i]; }
    return c;
}

inline int operator <= (int a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a <= b[i]; }
    return c;
}

inline int operator <= (const ivec2& a, int b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] <= b; }
    return c;
}



inline int operator > (const ivec2& a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] > b[i]; }
    return c;
}

inline int operator > (int a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a > b[i]; }
    return c;
}

inline int operator > (const ivec2& a, int b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] > b; }
    return c;
}




inline int operator >= (const ivec2& a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] >= b[i]; }
    return c;
}
inline int operator >= (int a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a >= b[i]; }
    return c;
}

inline int operator >= (const ivec2& a, int b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] >= b; }
    return c;
}

inline int operator != (const ivec2& a, const ivec2& b)
{
    int c = 1;
    for(int i = 0; i < 2;++i) { c = c && a[i] != b[i]; }
    return c;
}

//| unary op : componentwise
inline ivec2 operator - (const ivec2& a)
{
    ivec2 c;
    for(int i = 0; i < 2;++i) { c[i] = -a[i]; }
    return c;
}

/**
* @brief structure for 3-dimensional integer vector and its arithmetic.
*/
struct __declspec(dllexport) ivec3 {
    int v[3];
    static const int n;
    
    inline ivec3() { };
    
    inline ivec3(int a)
    {
	 v[1 - 1] = a;  v[2 - 1] = a;  v[3 - 1] = a;  
    }
    
    inline ivec3(int v_1  , int v_2  , int v_3 )
    {
	 v[1 - 1] = v_1;  v[2 - 1] = v_2;  v[3 - 1] = v_3; 
    }

    inline ivec3(const ivec3& a)
    {
	 v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1]; 
    }

    inline ivec3& operator=(const ivec3& a)
    {
	 v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1]; 
	return *this;
    }


    inline int& operator[] (int i) { return v[i]; }
    inline const int&  operator[] (int i) const { return v[i]; }
    inline int& operator() (int i) { return v[i % 3]; }
    inline const int&  operator() (int i) const { return v[i % 3]; }
};
//| binary op : componentwise

inline ivec3 operator + (const ivec3& a, const ivec3& b)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = a[i] + b[i]; }
    return c;
}

inline ivec3 operator + (int a, const ivec3& b)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = a + b[i]; }
    return c;
}

inline ivec3 operator + (const ivec3& a, int b)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = a[i] + b; }
    return c;
}



inline ivec3 operator - (const ivec3& a, const ivec3& b)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = a[i] - b[i]; }
    return c;
}

inline ivec3 operator - (int a, const ivec3& b)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = a - b[i]; }
    return c;
}

inline ivec3 operator - (const ivec3& a, int b)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = a[i] - b; }
    return c;
}



inline ivec3 operator * (const ivec3& a, const ivec3& b)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = a[i] * b[i]; }
    return c;
}

inline ivec3 operator * (int a, const ivec3& b)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = a * b[i]; }
    return c;
}

inline ivec3 operator * (const ivec3& a, int b)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = a[i] * b; }
    return c;
}

//
//
//inline ivec3 operator / (const ivec3& a, const ivec3& b)
//{
//    ivec3 c;
//    for(int i = 0; i < 3;++i) { c[i] = a[i] / b[i]; }
//    return c;
//}
//
//inline ivec3 operator / (int a, const ivec3& b)
//{
//    ivec3 c;
//    for(int i = 0; i < 3;++i) { c[i] = a / b[i]; }
//    return c;
//}
//
//inline ivec3 operator / (const ivec3& a, int b)
//{
//    ivec3 c;
//    for(int i = 0; i < 3;++i) { c[i] = a[i] / b; }
//    return c;
//}



//| cumulative op : componentwise


inline ivec3 operator += (ivec3& a, const ivec3& b)
{
    return a = (a + b);
}

inline ivec3 operator += (ivec3& a, int b)
{
    return a = (a + b);
}



inline ivec3 operator -= (ivec3& a, const ivec3& b)
{
    return a = (a - b);
}

inline ivec3 operator -= (ivec3& a, int b)
{
    return a = (a - b);
}



inline ivec3 operator *= (ivec3& a, const ivec3& b)
{
    return a = (a * b);
}

inline ivec3 operator *= (ivec3& a, int b)
{
    return a = (a * b);
}


//
//inline ivec3 operator /= (ivec3& a, const ivec3& b)
//{
//    return a = (a / b);
//}
//
//inline ivec3 operator /= (ivec3& a, int b)
//{
//    return a = (a / b);
//}



//| logical op : componentwise


inline int operator == (const ivec3& a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] == b[i]; }
    return c;
}

inline int operator == (int a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a == b[i]; }
    return c;
}

inline int operator == (const ivec3& a, int b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] == b; }
    return c;
}



inline int operator < (const ivec3& a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] < b[i]; }
    return c;
}

inline int operator < (int a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a < b[i]; }
    return c;
}

inline int operator < (const ivec3& a, int b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] < b; }
    return c;
}



inline int operator <= (const ivec3& a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] <= b[i]; }
    return c;
}

inline int operator <= (int a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a <= b[i]; }
    return c;
}

inline int operator <= (const ivec3& a, int b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] <= b; }
    return c;
}



inline int operator > (const ivec3& a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] > b[i]; }
    return c;
}

inline int operator > (int a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a > b[i]; }
    return c;
}

inline int operator > (const ivec3& a, int b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] > b; }
    return c;
}



inline int operator >= (const ivec3& a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] >= b[i]; }
    return c;
}

inline int operator >= (int a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a >= b[i]; }
    return c;
}

inline int operator >= (const ivec3& a, int b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] >= b; }
    return c;
}

inline int operator != (const ivec3& a, const ivec3& b)
{
    int c = 1;
    for(int i = 0; i < 3;++i) { c = c && a[i] != b[i]; }
    return c;
}

//| unary op : componentwise
inline ivec3 operator - (const ivec3& a)
{
    ivec3 c;
    for(int i = 0; i < 3;++i) { c[i] = -a[i]; }
    return c;
}

/**
* @brief structure for 4-dimensional integer vector and its arithmetic.
*/
struct __declspec(dllexport) ivec4 {
    int v[4];
    static const int n;
    
    inline ivec4() { };
    
    inline ivec4(int a)
    {
	 v[1 - 1] = a;  v[2 - 1] = a;  v[3 - 1] = a;  v[4 - 1] = a;  
    }
    
    inline ivec4(int v_1  , int v_2  , int v_3  , int v_4 )
    {
	 v[1 - 1] = v_1;  v[2 - 1] = v_2;  v[3 - 1] = v_3;  v[4 - 1] = v_4; 
    }

    inline ivec4(const ivec4& a)
    {
	 v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1];  v[4 - 1] = a[4 - 1]; 
    }

    inline ivec4& operator=(const ivec4& a)
    {
	 v[1 - 1] = a[1 - 1];  v[2 - 1] = a[2 - 1];  v[3 - 1] = a[3 - 1];  v[4 - 1] = a[4 - 1]; 
	return *this;
    }


    inline int& operator[] (int i) { return v[i]; }
    inline const int&  operator[] (int i) const { return v[i]; }
    inline int& operator() (int i) { return v[i % 4]; }
    inline const int&  operator() (int i) const { return v[i % 4]; }
};




//| binary op : componentwise


inline ivec4 operator + (const ivec4& a, const ivec4& b)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = a[i] + b[i]; }
    return c;
}

inline ivec4 operator + (int a, const ivec4& b)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = a + b[i]; }
    return c;
}

inline ivec4 operator + (const ivec4& a, int b)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = a[i] + b; }
    return c;
}



inline ivec4 operator - (const ivec4& a, const ivec4& b)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = a[i] - b[i]; }
    return c;
}

inline ivec4 operator - (int a, const ivec4& b)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = a - b[i]; }
    return c;
}

inline ivec4 operator - (const ivec4& a, int b)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = a[i] - b; }
    return c;
}



inline ivec4 operator * (const ivec4& a, const ivec4& b)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = a[i] * b[i]; }
    return c;
}

inline ivec4 operator * (int a, const ivec4& b)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = a * b[i]; }
    return c;
}

inline ivec4 operator * (const ivec4& a, int b)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = a[i] * b; }
    return c;
}


//
//inline ivec4 operator / (const ivec4& a, const ivec4& b)
//{
//    ivec4 c;
//    for(int i = 0; i < 4;++i) { c[i] = a[i] / b[i]; }
//    return c;
//}
//
//inline ivec4 operator / (int a, const ivec4& b)
//{
//    ivec4 c;
//    for(int i = 0; i < 4;++i) { c[i] = a / b[i]; }
//    return c;
//}
//
//inline ivec4 operator / (const ivec4& a, int b)
//{
//    ivec4 c;
//    for(int i = 0; i < 4;++i) { c[i] = a[i] / b; }
//    return c;
//}



//| cumulative op : componentwise


inline ivec4 operator += (ivec4& a, const ivec4& b)
{
    return a = (a + b);
}

inline ivec4 operator += (ivec4& a, int b)
{
    return a = (a + b);
}



inline ivec4 operator -= (ivec4& a, const ivec4& b)
{
    return a = (a - b);
}

inline ivec4 operator -= (ivec4& a, int b)
{
    return a = (a - b);
}



inline ivec4 operator *= (ivec4& a, const ivec4& b)
{
    return a = (a * b);
}

inline ivec4 operator *= (ivec4& a, int b)
{
    return a = (a * b);
}



//inline ivec4 operator /= (ivec4& a, const ivec4& b)
//{
//    return a = (a / b);
//}
//
//inline ivec4 operator /= (ivec4& a, int b)
//{
//    return a = (a / b);
//}



//| logical op : componentwise


inline int operator == (const ivec4& a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] == b[i]; }
    return c;
}

inline int operator == (int a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a == b[i]; }
    return c;
}

inline int operator == (const ivec4& a, int b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] == b; }
    return c;
}



inline int operator < (const ivec4& a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] < b[i]; }
    return c;
}

inline int operator < (int a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a < b[i]; }
    return c;
}

inline int operator < (const ivec4& a, int b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] < b; }
    return c;
}



inline int operator <= (const ivec4& a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] <= b[i]; }
    return c;
}

inline int operator <= (int a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a <= b[i]; }
    return c;
}

inline int operator <= (const ivec4& a, int b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] <= b; }
    return c;
}



inline int operator > (const ivec4& a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] > b[i]; }
    return c;
}

inline int operator > (int a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a > b[i]; }
    return c;
}

inline int operator > (const ivec4& a, int b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] > b; }
    return c;
}



inline int operator >= (const ivec4& a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] >= b[i]; }
    return c;
}

inline int operator != (const ivec4& a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] != b[i]; }
    return c;
}

inline int operator >= (int a, const ivec4& b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a >= b[i]; }
    return c;
}

inline int operator >= (const ivec4& a, int b)
{
    int c = 1;
    for(int i = 0; i < 4;++i) { c = c && a[i] >= b; }
    return c;
}



//| unary op : componentwise
inline ivec4 operator - (const ivec4& a)
{
    ivec4 c;
    for(int i = 0; i < 4;++i) { c[i] = -a[i]; }
    return c;
}

}; //namespace oph

#endif // !__ivec_h
