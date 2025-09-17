//
// Created by Alex on 15.09.2025.
//

#ifndef STARTER_H
#define STARTER_H
#include "../util/logging.hpp"
#include "../util/types/types.hpp"
#include "controllers/api/DatasetController.hpp"
#include "controllers/api/TransformationController.hpp"
#include "../service/TransformationService.hpp"

#include "internal/RestServer.hpp"


class Starter {

public:
    static i32 run(const u16 port = 8080, const u16 threads = 4, const std::string& address = "0.0.0.0") {
        Log::Logger().message(R"(
Starting...)");
        auto const _address = net::ip::make_address(address);

        net::io_context ioc{threads};

        auto router = std::make_shared<Router>();

        auto datasetService = std::make_shared<DatasetService>();
        auto transformationService = std::make_shared<TransformationService>();

        router->addController<DatasetController>(datasetService);
        router->addController<TransformationController>(datasetService, transformationService);

        std::make_shared<RestServer>(router, ioc, tcp::endpoint{_address, port})->run();

        Log::Logger().message(R"(
Server is running.)");

        Log::Logger().withColor(Log::Colors::Magenta,
            R"(

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ███╗   ██╗  ███████╗  ██╗   ██╗  ██████╗    ██████╗
        ████╗  ██║  ██╔════╝  ██║   ██║  ██╔══██╗  ██╔═══██╗
        ██╔██╗ ██║  █████╗    ██║   ██║  ██████╔╝  ██║   ██║
        ██║╚██╗██║  ██╔══╝    ██║   ██║  ██╔══██╗  ██║   ██║
        ██║ ╚████║  ███████╗  ╚██████╔╝  ██║  ██║  ╚██████╔╝
        ╚═╝  ╚═══╝  ╚══════╝   ╚═════╝   ╚═╝  ╚═╝   ╚═════╝

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|| Started with {} threads on: )", threads);

        Log::Logger().withColor(Log::Colors::Cyan, R"(http://{}:{})", address, port);
        Log::Logger().withColor(Log::Colors::Magenta, R"(

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

)");

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
