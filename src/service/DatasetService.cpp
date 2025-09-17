//
// Created by Alex on 15.09.2025.
//

#include "DatasetService.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include "../util/constants.hpp"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include "../util/logging.hpp"

namespace fs = std::filesystem;


std::vector<std::string> DatasetService::listAvailableDatasets(const std::string& directoryPath) const {
    std::vector<std::string> fileList;
    if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
        return fileList;
    }

    fs::recursive_directory_iterator it(directoryPath);
    fs::recursive_directory_iterator endit;

    while (it != endit) {
        if (fs::is_regular_file(*it) && it->path().extension() == ".csv") {
            fileList.push_back(it->path().string());
        }
        ++it;
    }
    return fileList;
}

std::string DatasetService::loadDataset(const std::string& filePath) {
    // лочим мьютекс на время всей операции
    std::lock_guard lock(_mutex);

    // парсим CSV
    auto [headers, rawData] = parseCsv(filePath);

    // генерируем уникальный ID
    const boost::uuids::uuid uuid = boost::uuids::random_generator()();
    std::string id = to_string(uuid);

    // cоздаем и заполняем Dataset
    auto dataset = std::make_shared<Dataset>();
    dataset->id = id;
    dataset->name = fs::path(filePath).filename().string();
    dataset->headers = std::move(headers);
    dataset->rawData = std::move(rawData);
    dataset->rowCount = dataset->rawData.size();
    dataset->columnCount = dataset->headers.size();
    dataset->createdAt = std::chrono::system_clock::now();

    // сейвим в мапу
    _datasets[id] = dataset;

    return id;
}

std::vector<std::pair<std::string, std::string>> DatasetService::loadedDatasetsList() const {
    std::lock_guard lock(_mutex);
    std::vector<std::pair<std::string, std::string>> list;
    list.reserve(_datasets.size());
    for (const auto& [id, datasetPtr] : _datasets) {
        list.emplace_back(id, datasetPtr->name);
    }
    return list;
}

std::vector<std::pair<std::string, std::string>> DatasetService::unloadDataset(const std::string& id) {
     std::lock_guard lock(_mutex);
     if(_datasets.contains(id)){
         _datasets.erase(id);
     }

     return loadedDatasetsList();
}


std::optional<PaginatedData> DatasetService::getDatasetPage(const std::string& datasetId, u32 page, u32 pageSize) const {
    std::lock_guard lock(_mutex);

    auto it = _datasets.find(datasetId);
    if (it == _datasets.end()) {
        return std::nullopt;
    }

    auto dataset = it->second;

    if (page < 1 || pageSize < 1) {
        // невалидные параметры пагинации
        return std::nullopt;
    }

    auto paginated = PaginatedData{};
    paginated.id = dataset->id;
    paginated.name = dataset->name;
    paginated.page = page;
    paginated.pageSize = pageSize;
    paginated.totalRows = dataset->rowCount;
    paginated.headers = dataset->headers;
    paginated.totalPages = static_cast<u32>(std::ceil(static_cast<double>(dataset->rowCount) / pageSize));

    if (paginated.totalPages == 0) paginated.totalPages = 1;


    // считаем диапазон строк для текущей страницы
    size_t startIndex = (page - 1) * pageSize;
    if (startIndex >= dataset->rowCount) {
        // страница находится за пределами данных, возвращаем пустые данные
        return paginated;
    }
    size_t endIndex = std::min(startIndex + pageSize, dataset->rowCount);

    // копируем нужный срез данных
    paginated.data.reserve(endIndex - startIndex);
    for(size_t i = startIndex; i < endIndex; ++i) {
        paginated.data.push_back(dataset->rawData[i]);
    }

    return paginated;
}

// парсер
std::pair<std::vector<std::string>, std::vector<std::vector<std::string>>>
DatasetService::parseCsv(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> data;
    std::string line;

    auto parseLine = [](const std::string& lineStr) {
        std::vector<std::string> row;
        std::string cell;
        bool inQuotes = false;

        for (size_t i = 0; i < lineStr.length(); ++i) {
            char c = lineStr[i];

            if (inQuotes) {
                if (c == '"') {
                    // проверяем, это экранированная кавычка или конец поля
                    if (i + 1 < lineStr.length() && lineStr[i+1] == '"') {
                        cell += '"'; // если это экранированная кавычка, добавляем одну в ячейку
                        i++; // пропускаем следующую кавычку
                    } else {
                        inQuotes = false; // иначе выходим из кавычек
                    }
                } else {
                    cell += c; // если это обычный символ внутри кавычек
                }
            } else { // не в кавычках
                if (c == '"') {
                    inQuotes = true;
                } else if (c == ',') {
                    row.push_back(cell);
                    cell.clear();
                } else {
                    cell += c;
                }
            }
        }
        row.push_back(cell); // добавляем последнюю ячейку
        return row;
    };

    // чтение заголовков
    if (std::getline(file, line)) {
        // убираем возможный символ возврата каретки
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        headers = parseLine(line);
    } else {
        throw std::runtime_error("CSV file is empty or header is missing.");
    }

    // чтение данных
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        std::vector<std::string> row = parseLine(line);

        // проверка на случай, если в данных больше колонок, чем в заголовке
        if (row.size() > headers.size()) {

            Log::Logger().warning("Data have more column than header!");

            row.resize(headers.size());
        }

        data.push_back(row);
    }

    return {headers, data};
}