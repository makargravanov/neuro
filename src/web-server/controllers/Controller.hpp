//
// Created by Alex on 15.09.2025.
//

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio.hpp>


namespace beast = boost::beast;
namespace http  = beast::http;
namespace net   = boost::asio;
using     tcp   = boost::asio::ip::tcp;

class Controller {

public:
     http::response<http::string_body> handleRequest(const http::request<http::string_body>& req) {
        if (req.method() == http::verb::get && req.target() == "/api/hello/") {
            http::response<http::string_body> res{http::status::ok, req.version()};
            res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
            res.set(http::field::content_type, "application/json");
            res.body() = R"({"response":"Hello"})";
            res.prepare_payload();
            return res;
        }
        return notFound();
    }
private:
    static http::response<http::string_body> mockService() {
        http::response<http::string_body> res{http::status::ok, 11};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "application/json");
        res.body() = R"([{"id":1,"name":"Item 1"},{"id":2,"name":"Item 2"}])";
        res.prepare_payload();
        return res;
    }

    static http::response<http::string_body> notFound() {
        http::response<http::string_body> res{http::status::not_found, 11};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "text/plain");
        res.body() = "The resource was not found.";
        res.prepare_payload();
        return res;
    }
};



#endif //CONTROLLER_H
