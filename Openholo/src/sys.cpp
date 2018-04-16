#include "sys.h"
#include <stdarg.h>
#include <stdio.h>

static FILE *fp;
 
void file_log(const char *fmt, ...)
{
#ifdef __CSKETCH_DEBUG
	if (fp == 0) {
		fp = fopen("my_log.txt", "w+t");
	}
    va_list ap;

    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
	fflush(fp);
#endif

}

#ifdef _MAC_OS
void * memalign(size_t align, size_t sz)
{
    /* this a hack so that runlog can link with mallocs that don't have */
    /* memalign */
    void *p = malloc(sz + align + 1);
    size_t d = (unsigned long)p % align;

    if (d != 0)
        p = (char *)p + align - d;

    return p;
}
#endif

FILE* file_read_open(const WChar* fname)
{
#ifdef _WIN32
	return _wfopen(fname, L"r+b");
#else
	return fopen(fname, "r+b");
#endif
}

FILE* file_write_open(const WChar* fname)
{
#ifdef _WIN32
	return _wfopen(fname, L"w+b");
#else
	return fopen(fname, "w+b");
#endif
}

FILE* file_read_open(const WChar* fname, const WChar* mode)
{
#ifdef _WIN32
	return _wfopen(fname, mode);
#else
	return fopen(fname, mode);
#endif
}

FILE* file_write_open(const WChar* fname, const WChar* mode)
{
#ifdef _WIN32
	return _wfopen(fname, mode);
#else
	return fopen(fname, mode);
#endif
}

WChar* string_cpy(WChar* dest, const WChar* src)
{
#ifdef _WIN32
	return wcscpy(dest, src);
#else
	return strcpy(dest, src);
#endif
}

WChar* string_cat(WChar* dest, const WChar* src)
{
#ifdef _WIN32
	return wcscat(dest, src);
#else
	return strcat(dest, src);
#endif
}

int string_cmp(const WChar* str1, const WChar* str2)
{
#ifdef _WIN32
	return wcscmp(str1, str2);
#else
	return strcmp(str1, str2);
#endif
}

