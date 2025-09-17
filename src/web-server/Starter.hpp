//
// Created by Alex on 15.09.2025.
//

#ifndef STARTER_H
#define STARTER_H
#include "../util/types/types.hpp"
#include "controllers/api/DatasetController.hpp"

#include "internal/RestServer.hpp"


class Starter {

public:
    static i32 run(const u16 port = 8080, const u16 threads = 4) {
        auto const address = net::ip::make_address("0.0.0.0");

        net::io_context ioc{threads};

        Router router{};

        auto datasetService = std::make_shared<DatasetService>();

        router.addController<DatasetController>(DatasetController(datasetService));

        std::make_shared<RestServer>(router, ioc, tcp::endpoint{address, port})->run();

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
