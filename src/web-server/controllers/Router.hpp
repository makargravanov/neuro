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

        for (const auto& route : controller->getRoutes()) {
            const std::string& path = route.getPathTemplate();
            if (_registeredPaths.contains(path)) {
                throw std::logic_error("Duplicate route path registered: " + path);
            }
            _registeredPaths.insert(path);
        }
        _controllers.push_back(std::move(controller));
    }

    http::response<http::string_body> handleRequest(const http::request<http::string_body>& req) {
        std::string_view targetPath = req.target();

        // убираем query-параметры из пути для сопоставления
        auto queryPos = targetPath.find('?');
        if (queryPos != std::string_view::npos) {
            targetPath = targetPath.substr(0, queryPos);
        }

        for (const auto& controller : _controllers) {
            for (const auto& route : controller->getRoutes()) {
                std::smatch match;
                std::string target_str(targetPath);

                // пытаемся сопоставить путь запроса с regex'ом роута
                if (std::regex_match(target_str, match, route.getPathRegex())) {

                    // путь совпал, теперь проверяем HTTP-метод
                    if (!route.methods.contains(req.method())) {
                        return IController::methodNotAllowed();
                    }

                    // собираем контекст запроса
                    RequestCtx ctx{req};
                    const auto& paramNames = route.getParamNames();
                    for (size_t i = 0; i < paramNames.size(); ++i) {
                        // match[0] - вся строка, match[1] - первая группа и т.д.
                        ctx.pathParams[paramNames[i]] = match[i + 1].str();
                    }

                    // вызываем нужный метод контроллера
                    switch (req.method()) {
                        case http::verb::get:     return controller->get(ctx);
                        case http::verb::post:    return controller->post(ctx);
                        case http::verb::patch:   return controller->patch(ctx);
                        case http::verb::delete_: return controller->delete_(ctx);
                        default:                  return IController::methodNotAllowed();
                    }
                }
            }
        }
        return IController::notFound("Route not found");
    }
};

#endif //CONTROLLER_H
