// src/util/ImageDatasetLoader.hpp
#ifndef IMAGEDATASETLOADER_HPP
#define IMAGEDATASETLOADER_HPP

#include <vector>
#include <string>
#include <filesystem>
#include <expected>
#include "types/eigen_types.hpp"

struct Error {
    std::string message;
};

struct ImgDataset {
    std::vector<Eigen::VectorXf> inputs;
    std::vector<Eigen::VectorXf> outputs;
    std::vector<std::string> classNames; // Сохраним имена классов для удобства
    u32 imageWidth{};
    u32 imageHeight{};
    u32 imageChannels{};
};

// Конфигурация для загрузки
struct ImageLoaderConfig {
    u32 targetWidth;
    u32 targetHeight;
    // Можно добавить выбор стратегии: u32 channels = 3 (форсировать RGB)
    u32 forceChannels = 0; // 0 - как в файле, 1 - Grey, 3 - RGB, 4 - RGBA
};

class ImageDatasetLoader {
    static std::vector<u8> resizeImage(const u8* inputData, int w, int h, int c, u32 targetW, u32 targetH);

public:
    static std::expected<ImgDataset, Error> loadFromDirectory(
        const std::filesystem::path& directoryPath,
        ImageLoaderConfig config
    );
};


#endif //IMAGEDATASETLOADER_HPP