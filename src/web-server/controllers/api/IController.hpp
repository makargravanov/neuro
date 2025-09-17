//
// Created by Alex on 16.09.2025.
//

#ifndef ICONTROLLER_H
#define ICONTROLLER_H
#include <set>
#include <string>
#include <nlohmann/json.hpp>

#include "../../server_types.h"
#include "../Route.hpp"

struct RequestCtx {
    const http::request<http::string_body>& originalRequest;
    std::map<std::string, std::string> pathParams;
};

using Handler = std::function<http::response<http::string_body>(const RequestCtx&)>;

struct RouteHandler {
    Route route;
    Handler handler;
};

class IController {
protected:
    std::vector<RouteHandler> _routeHandlers;

public:
    virtual ~IController() = default;

    explicit IController(std::vector<RouteHandler> handlers)
        : _routeHandlers(std::move(handlers)) {}

    // геттер для роутера, чтобы он мог проверить уникальность путей
    [[nodiscard]] const std::vector<RouteHandler>& getRouteHandlers() const {
        return _routeHandlers;
    }

    // единственный метод, который вызывает Router
    // он сам найдет нужный обработчик и вызовет его.
    http::response<http::string_body> dispatch(const Route& matchedRoute, const RequestCtx& ctx) {
        // роутер уже проверил совпадение метода, эта проверка для дополнительной надежности.
        if (!matchedRoute.methods.contains(ctx.originalRequest.method())) {
            return methodNotAllowed();
        }

        // находим нужный handler по шаблону пути.
        // так как Router уже нашел совпадение, мы уверены, что handler здесь есть.
        for (const auto&[route, handler] : _routeHandlers) {
            if (route.getPathTemplate() == matchedRoute.getPathTemplate() && route.methods == matchedRoute.methods) {
                return handler(ctx); // вызываем связанный обработчик
            }
        }

        return internalError("Dispatcher failed to find a handler for a matched route.");
    }

    static http::response<http::string_body> notFound(const std::string& reason) {
        http::response<http::string_body> res{http::status::not_found, 11};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "application/json");
        res.body() = R"({"error": ")" + reason + "\"}";
        res.prepare_payload();
        return res;
    }

    static http::response<http::string_body> methodNotAllowed() {
        http::response<http::string_body> res{http::status::method_not_allowed, 11};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "application/json");
        res.body() = R"({"error": "Method Not Allowed"})";
        res.prepare_payload();
        return res;
    }

    static http::response<http::string_body> internalError(const std::string& reason) {
        http::response<http::string_body> res{http::status::internal_server_error, 11};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "application/json");
        res.body() = reason;
        res.prepare_payload();
        return res;
    }

    static http::response<http::string_body> createJsonResponse(const http::status status, const nlohmann::json& body) {
        http::response<http::string_body> res{status, 11};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "application/json");
        if (!body.is_null() && !body.empty()) {
            res.body() = body.dump(); // .dump() сериализует JSON в строку
        }
        res.prepare_payload();
        return res;
    }

    static http::response<http::string_body> createErrorResponse(const http::status status, const std::string& message) {
        const nlohmann::json errorBody = {{"error", message}};
        return createJsonResponse(status, errorBody);
    }

    static std::map<std::string, std::string> parseQueryString(const std::string_view url) {
        std::map<std::string, std::string> params;
        const auto queryPos = url.find('?');
        if (queryPos == std::string_view::npos) {
            return params;
        }

        const std::string_view query = url.substr(queryPos + 1);
        std::string_view::size_type currentPos = 0;

        while(currentPos < query.size()) {
            const auto pairEndPos = query.find('&', currentPos);
            std::string_view pair = query.substr(currentPos, pairEndPos - currentPos);

            if (const auto valuePos = pair.find('='); valuePos != std::string_view::npos) {
                std::string key(pair.substr(0, valuePos));
                const std::string value(pair.substr(valuePos + 1));
                params[key] = value;
            }

            if (pairEndPos == std::string_view::npos) break;
            currentPos = pairEndPos + 1;
        }
        return params;
    }

};
#endif //ICONTROLLER_H