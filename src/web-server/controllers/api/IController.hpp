//
// Created by Alex on 16.09.2025.
//

#ifndef ICONTROLLER_H
#define ICONTROLLER_H
#include <set>
#include <string>

#include "../../server_types.h"
#include "../Route.hpp"


struct RequestCtx {
    const http::request<http::string_body>& originalRequest;
    std::map<std::string, std::string> pathParams;
};

class IController {
protected:
    std::set<Route> _routes;

public:
    virtual ~IController() = default;

    explicit IController(std::set<Route> routes) : _routes(std::move(routes)) {}

    [[nodiscard]] const std::set<Route>& getRoutes() const {
        return _routes;
    }

    virtual http::response<http::string_body> get(const RequestCtx& ctx) {
        return methodNotAllowed();
    }
    virtual http::response<http::string_body> post(const RequestCtx& ctx) {
        return methodNotAllowed();
    }
    virtual http::response<http::string_body> patch(const RequestCtx& ctx) {
        return methodNotAllowed();
    }
    virtual http::response<http::string_body> delete_(const RequestCtx& ctx) {
        return methodNotAllowed();
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

};
#endif //ICONTROLLER_H
