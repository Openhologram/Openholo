#pragma once
#include <stdlib.h>


class StringConverter
{
private:
	StringConverter();
	~StringConverter();
	static StringConverter *instance;
	static void Destroy() {
		delete instance;
	}
public:
	static StringConverter* getInstance() {
		if (instance == nullptr) {
			instance = new StringConverter();
			atexit(Destroy);
		}
		return instance;
	}

	wchar_t* M2W(char *input);
	char* W2M(wchar_t *input);
	
};

