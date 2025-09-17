//
// Created by Alex on 17.09.2025.
//

#ifndef COLORS_HPP
#define COLORS_HPP

#include <string_view>

namespace Log::Colors {
    constexpr std::string_view Reset   = "\x1b[0m";
    constexpr std::string_view Red     = "\x1b[31m";
    constexpr std::string_view Green   = "\x1b[32m";
    constexpr std::string_view Yellow  = "\x1b[33m";
    constexpr std::string_view Blue    = "\x1b[34m";
    constexpr std::string_view Magenta = "\x1b[35m";
    constexpr std::string_view Cyan    = "\x1b[36m";
    constexpr std::string_view White   = "\x1b[37m";
    constexpr std::string_view BoldRed = "\x1b[1;31m";
}

#endif //COLORS_HPP
