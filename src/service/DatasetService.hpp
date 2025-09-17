//
// Created by Alex on 15.09.2025.
//

#ifndef DATASETSERVICE_H
#define DATASETSERVICE_H
#include <chrono>
#include <shared_mutex>
#include <string>
#include <vector>
#include "../util/types/eigen_types.hpp"


struct Dataset {
    std::string id;
    std::string name; //имя файла
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> rawData;
    size_t rowCount = 0;
    size_t columnCount = 0;
    std::chrono::system_clock::time_point createdAt;
};

struct PaginatedData {
    std::string id;
    std::string name;
    u32 page;
    u32 pageSize;
    u32 totalPages;
    size_t totalRows;
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> data;
};


class DatasetService {
public:
    DatasetService() = default;

    /**
     * @brief Сканирует директорию и возвращает список имен .csv файлов.
     * @param directoryPath Путь к директории для сканирования, относительно исполняемого файла.
     * @return Вектор с именами файлов, включающими в себя полный путь. Например, сканируя папку csv/datasets мы нашли файл iris.csv, тогда среди имён будет csv/datasets/iris.csv
     */
    std::vector<std::string> listAvailableDatasets(const std::string& directoryPath) const;

    /**
     * @brief Загружает CSV-файл в память, присваивает ему уникальный ID.
     * @param filePath - полный путь к CSV-файлу, относительно исполняемого файла
     * @return Уникальный ID загруженного датасета.
     * @throws std::runtime_error если файл не найден или не удалось прочитать.
     */
    std::string loadDataset(const std::string& filePath);

    /**
     * @brief Отдаёт список пар id-имя для всех загруженных датасетов
     * @return Список пар id-имя для всех загруженных датасетов
     */
    std::vector<std::pair<std::string, std::string>> loadedDatasetsList() const;

    /**
     * @brief Выгружает датасет
     * @param id - id датасета, который надо выгрузить из памяти
     * @return Обновлённый список пар id-имя для всех загруженных датасетов
     */
    std::vector<std::pair<std::string, std::string>> unloadDataset(const std::string& id);

    /**
     * @brief Возвращает страницу данных из ранее загруженного датасета.
     * @param datasetId Уникальный ID датасета.
     * @param page Номер страницы (начиная с 1).
     * @param pageSize Количество записей на странице.
     * @return Структура PaginatedData или std::nullopt, если датасет не найден.
     */
    std::optional<PaginatedData> getDatasetPage(const std::string& datasetId, u32 page, u32 pageSize) const;

    /**
     * @brief Возвращает указатель на объект датасета по его ID.
     * @param id Уникальный ID датасета.
     * @return std::shared_ptr<const Dataset> или nullptr, если датасет не найден.
     */
    std::shared_ptr<const Dataset> getDatasetById(const std::string& id) const;

    /**
     * @brief Регистрирует новый, трансформированный датасет в памяти.
     * @param dataset Указатель на новый объект датасета.
     * @param datasetName Имя, которое будет присвоено новому датасету.
     * @return ID нового зарегистрированного датасета.
     */
    std::string registerTransformedDataset(std::shared_ptr<Dataset> dataset, const std::string& datasetName);

    /**
     * @brief Создает копию существующего датасета и регистрирует её под новым именем.
     * @param sourceId ID исходного датасета, который нужно скопировать.
     * @param newName Новое имя для скопированного датасета.
     * @return ID нового зарегистрированного датасета.
     * @throws std::runtime_error если исходный датасет не найден.
     */
    std::string copyAndRegisterDataset(const std::string& sourceId, const std::string& newName);

    /**
     * @brief Сохраняет данные датасета из памяти в новый CSV файл.
     * @param datasetId ID датасета, который нужно сохранить.
     * @param newName Имя для нового файла (без расширения).
     * @return Полный путь к созданному файлу.
     * @throws std::runtime_error если датасет не найден или не удалось записать файл.
     */
    std::string saveDatasetToFile(const std::string& datasetId, const std::string& newName);

private:
    static std::pair<std::vector<std::string>, std::vector<std::vector<std::string>>> parseCsv(const std::string& filePath);

    std::string registerTransformedDataset_locked(std::shared_ptr<Dataset> dataset, const std::string& datasetName);

    std::unordered_map<std::string, std::shared_ptr<Dataset>> _datasets{};

    // мьютекс для потокобезопасного доступа к _datasets
    // mutable позволяет использовать мьютекс в const-методах для безопасного чтения
    mutable std::shared_mutex _mutex;
};



#endif //DATASETSERVICE_H