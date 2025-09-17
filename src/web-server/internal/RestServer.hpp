//
// Created by Alex on 15.09.2025.
//

#ifndef RESTSERVER_H
#define RESTSERVER_H

#include "HttpSession.hpp"



class RestServer : public std::enable_shared_from_this<RestServer> {
    net::io_context& _ioc;
    tcp::acceptor _acceptor;
    Router _controller;

public:
    RestServer(net::io_context& ioc, tcp::endpoint endpoint)
        : _ioc(ioc), _acceptor(ioc) {
        beast::error_code ec;

        _acceptor.open(endpoint.protocol(), ec);
        if (ec) {
            fail(ec, "open");
            return;
        }

        _acceptor.set_option(net::socket_base::reuse_address(true), ec);
        if (ec) {
            fail(ec, "set_option");
            return;
        }

        _acceptor.bind(endpoint, ec);
        if (ec) {
            fail(ec, "bind");
            return;
        }

        _acceptor.listen(net::socket_base::max_listen_connections, ec);
        if (ec) {
            fail(ec, "listen");
            return;
        }
    }

    void run() {
        doAccept();
    }

private:
    void doAccept() {
        _acceptor.async_accept(
            net::make_strand(_ioc),
            beast::bind_front_handler(
                &RestServer::onAccept,
                shared_from_this()));
    }

    void onAccept(beast::error_code ec, tcp::socket socket) {
        if (ec) {
            fail(ec, "accept");
        }
        else {
            std::make_shared<HttpSession>(std::move(socket), _controller)->run();
        }

        doAccept();
    }
};


#endif //RESTSERVER_H
