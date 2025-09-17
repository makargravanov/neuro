//
// Created by Alex on 17.09.2025.
//

#include "TransformationService.hpp"


std::shared_ptr<Dataset> TransformationService::removeColumn(std::shared_ptr<const Dataset> source, const std::string& columnName) {
    // находим индекс колонки, которую нужно удалить
    auto it = std::ranges::find(source->headers, columnName);
    if (it == source->headers.end()) {
        throw std::runtime_error("Column '" + columnName + "' not found.");
    }
    const size_t columnIndexToRemove = std::distance(source->headers.begin(), it);

    // создаем новый объект датасета
    auto transformedDataset = std::make_shared<Dataset>();

    // копируем заголовки, пропуская удаляемый
    transformedDataset->headers.reserve(source->headers.size() - 1);
    for (size_t i = 0; i < source->headers.size(); ++i) {
        if (i != columnIndexToRemove) {
            transformedDataset->headers.push_back(source->headers[i]);
        }
    }

    // копируем данные, пропуская значения из удаляемой колонки
    transformedDataset->rawData.reserve(source->rawData.size());
    for (const auto& sourceRow : source->rawData) {
        std::vector<std::string> newRow;
        newRow.reserve(sourceRow.size() - 1);
        for (size_t i = 0; i < sourceRow.size(); ++i) {
            if (i != columnIndexToRemove) {
                newRow.push_back(sourceRow[i]);
            }
        }
        transformedDataset->rawData.push_back(newRow);
    }

    // обновляем метаданные
    transformedDataset->rowCount = transformedDataset->rawData.size();
    transformedDataset->columnCount = transformedDataset->headers.size();

    return transformedDataset;
}