//
// Created by Alex on 15.09.2025.
//

#ifndef STARTER_H
#define STARTER_H
#include "../util/types/types.hpp"

#include "internal/RestServer.hpp"


class Starter {

public:
    static i32 run(const u16 port = 8080, const u16 threads = 4) {
        auto const address = net::ip::make_address("0.0.0.0");

        net::io_context ioc{threads};

        std::make_shared<RestServer>(ioc, tcp::endpoint{address, port})->run();

        std::vector<std::thread> v;
        v.reserve(threads - 1);
        for (auto i = threads - 1; i > 0; --i)
            v.emplace_back(
                [&ioc]{
                    ioc.run();
                });
        ioc.run();

        return 0;
    }
};



#endif //STARTER_H
