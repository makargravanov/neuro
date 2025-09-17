//
// Created by Alex on 16.09.2025.
//

#ifndef DATASETCONTROLLER_H
#define DATASETCONTROLLER_H
#include <filesystem>
#include "IController.hpp"
#include "../../../service/DatasetService.hpp"
#include "../../../util/constants.hpp"

using StringResponse = http::response<http::string_body>;
using json = nlohmann::json;
namespace fs = std::filesystem;

// --- JSON Сериализация для наших структур данных ---
// Это специальная функция, которую nlohmann::json находит сам
// и использует для преобразования наших кастомных структур в JSON.
inline void to_json(json& j, const PaginatedData& p) {
    j = json{
        {"id", p.id},
        {"name", p.name},
        {"page", p.page},
        {"pageSize", p.pageSize},
        {"totalPages", p.totalPages},
        {"totalRows", p.totalRows},
        {"headers", p.headers},
        {"data", p.data}
    };
}


class DatasetController : public IController {
    std::shared_ptr<DatasetService> _datasetService;

public:
    explicit DatasetController(std::shared_ptr<DatasetService> service)
        : IController({
              // Получить список доступных .csv файлов на диске
              {
                  Route("/api/v1/datasets/available", {http::verb::get}),
                  [this](const RequestCtx& ctx) { return this->getAvailableDatasets(ctx); }
              },
              // Получить список уже загруженных в память датасетов
              {
                  Route("/api/v1/datasets/loaded", {http::verb::get}),
                  [this](const RequestCtx& ctx) { return this->getLoadedDatasets(ctx); }
              },
              // Загрузить датасет из файла в память
              {
                  Route("/api/v1/datasets/load", {http::verb::post}),
                  [this](const RequestCtx& ctx) { return this->loadNewDataset(ctx); }
              },
              // Получить страницу данных из загруженного датасета
              {
                  Route("/api/v1/datasets/{id}", {http::verb::get}),
                  [this](const RequestCtx& ctx) { return this->getDatasetPageById(ctx); }
              },
              // Выгрузить датасет из памяти
              {
                  Route("/api/v1/datasets/{id}", {http::verb::delete_}),
                  [this](const RequestCtx& ctx) { return this->unloadDatasetById(ctx); }
              },
              // Сохранить текущий датасет под новым именем
              {
                  Route("/api/v1/datasets/{id}/save-as", {http::verb::post}),
                  [this](const RequestCtx& ctx) { return this->handleSaveDatasetAs(ctx); }
              }
          }),
          _datasetService(std::move(service)) {
    }

private:
    http::response<http::string_body> getAvailableDatasets(const RequestCtx& ctx) {
        auto files = _datasetService->listAvailableDatasets(FRAMEWORK_CONSTANTS::datasetsDirectory);
        json responseBody = files;
        return createJsonResponse(http::status::ok, responseBody);
    }

    http::response<http::string_body> getLoadedDatasets(const RequestCtx& ctx) {
        auto loadedList = _datasetService->loadedDatasetsList();
        json responseBody = json::array();
        for (const auto& [id, name]: loadedList) {
            responseBody.push_back({{"id", id}, {"name", name}});
        }
        return createJsonResponse(http::status::ok, responseBody);
    }

    http::response<http::string_body> loadNewDataset(const RequestCtx& ctx) {
        try {
            json requestBody = json::parse(ctx.originalRequest.body());
            std::string filePath = requestBody.at("filePath").get<std::string>();

            std::string newId = _datasetService->loadDataset(filePath);

            json responseBody = {{"datasetId", newId}};
            return createJsonResponse(http::status::created, responseBody);
        } catch (const json::parse_error& e) {
            return createErrorResponse(http::status::bad_request, "Invalid JSON format: " + std::string(e.what()));
        } catch (const json::type_error& e) {
            return createErrorResponse(http::status::bad_request, "JSON type error: " + std::string(e.what()));
        } catch (const std::exception& e) {
            return createErrorResponse(http::status::internal_server_error, e.what());
        }
    }

    http::response<http::string_body> getDatasetPageById(const RequestCtx& ctx) {
        const auto& id = ctx.pathParams.at("id");
        auto queryParams = parseQueryString(ctx.originalRequest.target());

        u32 page = 1;
        u32 pageSize = 50;

        if (queryParams.contains("page")) {
            std::from_chars(queryParams["page"].data(), queryParams["page"].data() + queryParams["page"].size(), page);
        }
        if (queryParams.contains("pageSize")) {
            std::from_chars(queryParams["pageSize"].data(),
                            queryParams["pageSize"].data() + queryParams["pageSize"].size(), pageSize);
        }

        auto pageDataOpt = _datasetService->getDatasetPage(id, page, pageSize);
        if (!pageDataOpt) {
            return createErrorResponse(http::status::not_found, "Dataset with id '" + id + "' not found.");
        }

        json responseBody = *pageDataOpt;
        return createJsonResponse(http::status::ok, responseBody);
    }

    http::response<http::string_body> unloadDatasetById(const RequestCtx& ctx) {
        const auto& id = ctx.pathParams.at("id");
        _datasetService->unloadDataset(id);
        http::response<http::string_body> res{http::status::no_content, ctx.originalRequest.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.prepare_payload();
        return res;
    }

    http::response<http::string_body> handleSaveDatasetAs(const RequestCtx& ctx) {
        try {
            const auto& sourceId = ctx.pathParams.at("id");
            json requestBody = json::parse(ctx.originalRequest.body());
            std::string newName = requestBody.at("newName").get<std::string>();

            // Шаг 1: Сохранить датасет в новый CSV файл
            std::string newFilePath = _datasetService->saveDatasetToFile(sourceId, newName);

            // Шаг 2: Загрузить этот новый файл в память, чтобы он стал доступен для работы
            std::string newDatasetId = _datasetService->loadDataset(newFilePath);

            // Получаем только имя файла для ответа
            std::string newDatasetName = fs::path(newFilePath).filename().string();

            json responseBody = {
                {"newDatasetId", newDatasetId},
                {"newDatasetName", newDatasetName}
            };
            return createJsonResponse(http::status::created, responseBody);
        } catch (const json::parse_error& e) {
            return createErrorResponse(http::status::bad_request, "Invalid JSON format: " + std::string(e.what()));
        } catch (const std::exception& e) {
            return createErrorResponse(http::status::bad_request, e.what());
        }
    }
};


#endif //DATASETCONTROLLER_H