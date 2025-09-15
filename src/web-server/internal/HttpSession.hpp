//
// Created by Alex on 15.09.2025.
//

#ifndef HTTPSESSION_H
#define HTTPSESSION_H

#include <iostream>

#include "../controllers/Controller.hpp"

inline void fail(beast::error_code ec, char const* what) {
    std::println(std::cerr, "{} : {}", what, ec.message());
}

class HttpSession : public std::enable_shared_from_this<HttpSession> {
    beast::tcp_stream _stream;
    beast::flat_buffer _buffer;
    http::request<http::string_body> _req;
    Controller& _apiController;

public:
    HttpSession(tcp::socket&& socket, Controller& controller)
        : _stream(std::move(socket)), _apiController(controller)
    {}

    void run() {
        doRead();
    }

private:
    void doRead() {
        _req = {};
        http::async_read(_stream, _buffer, _req,
            beast::bind_front_handler(
                &HttpSession::onRead,
                shared_from_this()));
    }

    void onRead(beast::error_code ec, std::size_t bytesTransferred) {
        boost::ignore_unused(bytesTransferred);

        if (ec == http::error::end_of_stream)
            return doClose();

        if (ec)
            return fail(ec, "read");

        handleRequest();
    }

    void handleRequest() {
        http::response<http::string_body> res = _apiController.handleRequest(_req);
        sendResponse(std::move(res));
    }

    void sendResponse(http::response<http::string_body>&& res) {
        auto sp = std::make_shared<http::response<http::string_body>>(std::move(res));

        http::async_write(_stream, *sp,
            [self = shared_from_this(), sp](beast::error_code ec, std::size_t bytes) {
                self->onWrite(ec, bytes, sp->need_eof());
            });
    }

    void onWrite(beast::error_code ec, std::size_t bytesTransferred, bool close) {
        boost::ignore_unused(bytesTransferred);

        if (ec)
            return fail(ec, "write");

        if (close) {
            return doClose();
        }

        doRead();
    }

    void doClose() {
        beast::error_code ec;
        _stream.socket().shutdown(tcp::socket::shutdown_send, ec);
    }
};



#endif //HTTPSESSION_H
