//
// Created by Alex on 17.09.2025.
//

#ifndef TRANSFORMATIONCONTROLLER_H
#define TRANSFORMATIONCONTROLLER_H




#include "IController.hpp"
#include "../../../service/DatasetService.hpp"
#include "../../../service/TransformationService.hpp"

using json = nlohmann::json;

class TransformationController : public IController {
    std::shared_ptr<DatasetService> _datasetService;
    std::shared_ptr<TransformationService> _transformationService;

public:
    explicit TransformationController(std::shared_ptr<DatasetService> ds, std::shared_ptr<TransformationService> ts)
        : IController({
              {
                  Route("/api/v1/datasets/{id}/transform/remove-column", {http::verb::post}),
                  [this](const RequestCtx& ctx) { return this->handleRemoveColumn(ctx); }
              }
          }),
          _datasetService(std::move(ds)),
          _transformationService(std::move(ts)) {}

private:
    http::response<http::string_body> handleRemoveColumn(const RequestCtx& ctx) {
        try {
            const auto& sourceId = ctx.pathParams.at("id");

            auto sourceDataset = _datasetService->getDatasetById(sourceId);
            if (!sourceDataset) {
                return createErrorResponse(http::status::not_found, "Source dataset not found.");
            }

            // парсим тело запроса, чтобы получить имя колонки
            json requestBody = json::parse(ctx.originalRequest.body());
            std::string columnName = requestBody.at("columnName").get<std::string>();

            // вызываем сервис для выполнения трансформации
            auto transformedDataset = _transformationService->removeColumn(sourceDataset, columnName);

            // регистрируем результат как новый датасет
            // Имя нового датасета по умолчанию будет таким же, как у исходного, с пометкой о трансформации
            std::string newDatasetName = sourceDataset->name + " (transformed)";
            std::string newId = _datasetService->registerTransformedDataset(transformedDataset, newDatasetName);

            // возвращаем ответ с информацией о новом датасете
            json responseBody = {
                {"newDatasetId", newId},
                {"newDatasetName", newDatasetName}
            };
            return createJsonResponse(http::status::ok, responseBody);

        } catch (const json::parse_error& e) {
            return createErrorResponse(http::status::bad_request, "Invalid JSON format: " + std::string(e.what()));
        } catch (const std::exception& e) {
            return createErrorResponse(http::status::bad_request, e.what());
        }
    }
};



#endif //TRANSFORMATIONCONTROLLER_H