#ifndef __sys_h
#define __sys_h
#ifdef _MSC_VER
#include <windows.h>
#include <conio.h>


#define LOG _cprintf
#elif __GNUC__
#include <unistd.h>
#include <stdio.h>


#define LOG printf
#endif
#endif