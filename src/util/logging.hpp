//
// Created by Alex on 17.09.2025.
//

#ifndef LOGGING_HPP
#define LOGGING_HPP
#include "constants.hpp"
#include <print>
#include <format>
#include <string_view>
#include <chrono>
#include <source_location>
#include <iostream>

#include "colors.hpp"


namespace Log {
    namespace detail {
        inline void printMessage(const std::source_location& loc, FRAMEWORK_CONSTANTS::LogLevel level,
                                 std::string_view message) {

            if (level > FRAMEWORK_CONSTANTS::runtimeLogLevel) return;

            const auto now = std::chrono::system_clock::now();
            const auto timeStr = std::format("{:%Y-%m-%d %H:%M:%S}", now);

            std::string_view levelColor = Colors::White;
            const char* levelStr = "UNKNOWN";

            switch (level) {
                case FRAMEWORK_CONSTANTS::LogLevel::LOG_ERROR:
                    levelStr = "ERROR";
                    levelColor = Colors::BoldRed;
                    break;
                case FRAMEWORK_CONSTANTS::LogLevel::LOG_WARNING:
                    levelStr = "WARNING";
                    levelColor = Colors::Yellow;
                    break;
                case FRAMEWORK_CONSTANTS::LogLevel::LOG_INFO:
                    levelStr = "INFO";
                    levelColor = Colors::Green;
                    break;
                case FRAMEWORK_CONSTANTS::LogLevel::LOG_DEBUG:
                    levelStr = "DEBUG";
                    levelColor = Colors::Cyan;
                    break;
                case FRAMEWORK_CONSTANTS::LogLevel::LOG_NONE:
                    return;
            }

            const std::string_view fileName = std::string_view(loc.file_name()).substr(std::string_view(loc.file_name()).find_last_of("/\\") + 1);

            std::println(std::cout, "[{}] [{}{:<7}{}] [{}:{}] [{}] {}{}{}",
                         timeStr,
                         levelColor, levelStr, Colors::Reset,
                         fileName,
                         loc.line(),
                         loc.function_name(),
                         levelColor, message, Colors::Reset);
        }
    }

    class Logger {
        std::source_location m_location;

    public:
        explicit Logger(const std::source_location& loc = std::source_location::current())
            : m_location(loc) {}

        template<typename... Args>
        void info(std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_INFO) {
                detail::printMessage(m_location, FRAMEWORK_CONSTANTS::LogLevel::LOG_INFO, std::format(formatStr, std::forward<Args>(args)...));
            }
        }


        template<typename... Args>
        void warning(std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_WARNING) {
                detail::printMessage(m_location, FRAMEWORK_CONSTANTS::LogLevel::LOG_WARNING, std::format(formatStr, std::forward<Args>(args)...));
            }
        }

        template<typename... Args>
        void error(std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_ERROR) {
                detail::printMessage(m_location, FRAMEWORK_CONSTANTS::LogLevel::LOG_ERROR, std::format(formatStr, std::forward<Args>(args)...));
            }
        }

        template<typename... Args>
        void debug(std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_DEBUG) {
                detail::printMessage(m_location, FRAMEWORK_CONSTANTS::LogLevel::LOG_DEBUG, std::format(formatStr, std::forward<Args>(args)...));
            }
        }

    };
}
#endif //LOGGING_HPP
