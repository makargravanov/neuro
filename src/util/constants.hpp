
#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP
#include <string>

namespace FRAMEWORK_CONSTANTS {

    enum class LogLevel {
        LOG_NONE,    // Логирование полностью отключено
        LOG_ERROR,   // Только ошибки, которые могут привести к падению или некорректной работе
        LOG_WARNING, // Предупреждения о потенциальных проблемах
        LOG_MESSAGE,
        LOG_INFO,    // Основная информация о ходе выполнения
        LOG_DEBUG    // Детальная отладочная информация
    };

    constexpr LogLevel compileTimeLogLevel = LogLevel::LOG_INFO;
    constexpr LogLevel runtimeLogLevel = compileTimeLogLevel;

    inline std::string datasetsDirectory = "datasets";
}

#endif