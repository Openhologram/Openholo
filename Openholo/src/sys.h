#ifndef __sys_h
#define __sys_h

#ifdef  _WIN32

#include <windows.h>
#include <conio.h>
#include <stdio.h>

typedef  wchar_t WChar;

#define LOG _cprintf
void file_log(const char *fmt, ...);

#endif

#ifdef	_MAC_OS

#include <stdio.h>

#define LOG printf

void file_log(const char *fmt, ...);

#include <unistd.h>
#include <signal.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#include <ucontext.h>

#include <string.h>
#include <wchar.h>

#endif // !__sys_h


//#ifdef _WIN32
//#include  "GL/glew.h"
//#include  <GL/gl.h>
//#include  <GL/glu.h>
//#include  <GL/glext.h>
//#endif


#ifdef _MAC_OS

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

typedef  char WChar;

#endif

#include <stdio.h>

#ifdef _LINUX

#include "graphics/unistd.h"
#include "graphics/values.h"
#define	LOG printf

#endif

#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>

#define FLOG fprintf


#ifdef _MAC_OS

void * memalign(size_t align, size_t sz);

#endif

FILE* file_read_open(const WChar* fname);
FILE* file_write_open(const WChar* fname);

FILE* file_read_open(const WChar* fname, const WChar* mode);
FILE* file_write_open(const WChar* fname, const WChar* mode);

WChar* string_cpy(WChar* dest, const WChar* src);
WChar* string_cat(WChar* dest, const WChar* src);
int    string_cmp(const WChar* str1, const WChar* str2);


#endif


//#ifndef	GL_RADEON
//#define GL_RADEON
//#endif
