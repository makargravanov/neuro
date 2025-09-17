//
// Created by Alex on 15.09.2025.
//

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "../server_types.h"
#include "api/IController.hpp"

class Router {
    std::vector<std::unique_ptr<IController>> _controllers;
    // Для быстрой проверки дубликатов путей при запуске
    std::set<std::string> _registeredPaths;

public:
    Router() = default;

    template<typename TController, typename... Args>
    void addController(Args&&... args) {
        auto controller = std::make_unique<TController>(std::forward<Args>(args)...);

        for (const auto& handler : controller->getRouteHandlers()) {
            const std::string& path = handler.route.getPathTemplate();
            for (const auto& method : handler.route.methods) {
                std::string uniqueIdentifier = std::string(http::to_string(method)) + ":" + path;
                if (_registeredPaths.contains(uniqueIdentifier)) {
                    throw std::logic_error("Duplicate route registered: " + uniqueIdentifier);
                }
                _registeredPaths.insert(uniqueIdentifier);
            }
        }
        _controllers.push_back(std::move(controller));
    }

    http::response<http::string_body> handleRequest(const http::request<http::string_body>& req) {
        std::string_view target_path = req.target();
        auto query_pos = target_path.find('?');
        if (query_pos != std::string_view::npos) {
            target_path = target_path.substr(0, query_pos);
        }

        bool path_matched = false;
        for (const auto& controller : _controllers) {
            for (const auto& routeHandler : controller->getRouteHandlers()) {
                std::smatch match;
                std::string target_str(target_path);

                if (std::regex_match(target_str, match, routeHandler.route.getPathRegex())) {
                    path_matched = true;
                    if (routeHandler.route.methods.contains(req.method())) {
                        // НАШЛИ ПОЛНОЕ СОВПАДЕНИЕ (путь + метод)!

                        // собираем контекст
                        RequestCtx ctx{req};
                        const auto& paramNames = routeHandler.route.getParamNames();
                        for (size_t i = 0; i < paramNames.size(); ++i) {
                            ctx.pathParams[paramNames[i]] = match[i + 1].str();
                        }

                        // передаем управление контроллеру
                        return controller->dispatch(routeHandler.route, ctx);
                    }
                }
            }
        }

        // если мы здесь - значит полного совпадения не найдено
        if (path_matched) {
            return IController::methodNotAllowed();
        }

        return IController::notFound("Route not found");
    }
};
#endif //CONTROLLER_H
