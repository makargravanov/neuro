//
// Created by Alex on 15.09.2025.
//

#include "bridge.h"
#include <print>
#include <boost/asio.hpp>

// Глобальный объект Asio, скрытый в этом .cpp файле
boost::asio::io_context g_io_context;

void initialize_network() {
    std::println("Network bridge initialized with Boost.Asio.");
    // Здесь может быть какая-то начальная настройка
}

void run_some_network_task() {
    std::println("Running a task via Boost.Asio...");
    // Пример использования: запустить обработчик
    g_io_context.run();
    std::println("Task finished.");
}