//
// Created by Alex on 17.09.2025.
//
module;

#ifdef _WIN32
#include <windows.h>
#endif

export module EnableColors;





export namespace Log::Platform {
    inline void enableColors() {
#ifdef _WIN32
        const HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut == INVALID_HANDLE_VALUE) {
            return;
        }

        DWORD dwMode = 0;
        if (!GetConsoleMode(hOut, &dwMode)) {
            return;
        }

        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if (!SetConsoleMode(hOut, dwMode)) {
            return;
        }
#endif
    }
}


