#include "StringConverter.h"
#include <string.h>
#include <stringapiset.h>

StringConverter* StringConverter::instance = nullptr;

StringConverter::StringConverter()
{
}


StringConverter::~StringConverter()
{
}



wchar_t* StringConverter::M2W(char *input)
{
	int nLen = MultiByteToWideChar(CP_ACP, 0, input, strlen(input), NULL, NULL);
	
}

char* StringConverter::W2M(wchar_t *input)
{


}