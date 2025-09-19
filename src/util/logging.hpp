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
                                 std::string_view message, bool includeFunctionName) {

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
                case FRAMEWORK_CONSTANTS::LogLevel::LOG_MESSAGE:
                    levelStr = "MESSAGE";
                    levelColor = Colors::Magenta;
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

            std::string functionPart;
            if (includeFunctionName) {
                functionPart = std::format("[{}] ", loc.function_name());
            }

            std::println(std::cout, "[{}] [{}{:<7}{}] [{}:{}] {}{}{}{}",
                         timeStr,
                         levelColor, levelStr, Colors::Reset,
                         fileName,
                         loc.line(),
                         functionPart,
                         levelColor, message, Colors::Reset);
        }

        inline void printMessageSimple(std::string_view color, std::string_view message) {
            std::print(std::cout, "{}{}{}", color, message, Colors::Reset);
        }
    }

    class Logger {
        std::source_location _location;
        bool _includeFunctionName = false;

    public:
        explicit Logger(const std::source_location& loc = std::source_location::current())
            : _location(loc) {}

        Logger& withFunction() {
            _includeFunctionName = true;
            return *this;
        }

        template<typename... Args>
        void info(std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_INFO) {
                detail::printMessage(_location, FRAMEWORK_CONSTANTS::LogLevel::LOG_INFO, std::format(formatStr, std::forward<Args>(args)...), _includeFunctionName);
            }
        }

        template<typename... Args>
        void message(std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_MESSAGE) {
                detail::printMessage(_location, FRAMEWORK_CONSTANTS::LogLevel::LOG_MESSAGE, std::format(formatStr, std::forward<Args>(args)...), _includeFunctionName);
            }
        }

        template<typename... Args>
        void warning(std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_WARNING) {
                detail::printMessage(_location, FRAMEWORK_CONSTANTS::LogLevel::LOG_WARNING, std::format(formatStr, std::forward<Args>(args)...), _includeFunctionName);
            }
        }

        template<typename... Args>
        void error(std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_ERROR) {
                detail::printMessage(_location, FRAMEWORK_CONSTANTS::LogLevel::LOG_ERROR, std::format(formatStr, std::forward<Args>(args)...), _includeFunctionName);
            }
        }

        template<typename... Args>
        void debug(std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_DEBUG) {
                detail::printMessage(_location, FRAMEWORK_CONSTANTS::LogLevel::LOG_DEBUG, std::format(formatStr, std::forward<Args>(args)...), _includeFunctionName);
            }
        }

        template<typename... Args>
        void withColor(std::string_view customColor, std::format_string<Args...> formatStr, Args&&... args) const {
            if constexpr (FRAMEWORK_CONSTANTS::compileTimeLogLevel >= FRAMEWORK_CONSTANTS::LogLevel::LOG_MESSAGE) {
                detail::printMessageSimple(customColor, std::format(formatStr, std::forward<Args>(args)...));
            }
        }
    };
}
#endif //LOGGING_HPP
