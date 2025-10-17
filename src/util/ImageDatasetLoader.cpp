// src/util/ImageDatasetLoader.cpp
#include "ImageDatasetLoader.hpp"
#include <map>
#include <print>
#include "logging.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

std::vector<u8> ImageDatasetLoader::resizeImage(const u8* inputData, int w, int h, int c, u32 targetW, u32 targetH) {
    std::vector<u8> outputData(targetW * targetH * c);

    // Используем stbir_resize_uint8_linear для быстрого и качественного изменения размера.
    // Он работает с sRGB, что подходит для фотографий.
    // 0 в stride означает, что данные идут плотно.
    stbir_resize_uint8_linear(inputData, w, h, 0,
                              outputData.data(), targetW, targetH, 0,
                              static_cast<stbir_pixel_layout>(c));

    return outputData;
}

std::expected<ImgDataset, Error> ImageDatasetLoader::loadFromDirectory(const std::filesystem::path& directoryPath, ImageLoaderConfig config) {
    if (!std::filesystem::exists(directoryPath)) {
        return std::unexpected(Error{std::format("Dataset directory does not exist: {}", directoryPath.string())});
    }

    ImgDataset dataset;
    dataset.imageWidth = config.targetWidth;
    dataset.imageHeight = config.targetHeight;

    std::map<std::string, u32> classMap;
    std::vector<std::string> classNamesList;
    u32 nextClassId = 0;

    Log::Logger().info("Scanning directory: {}", directoryPath.string());
    Log::Logger().info("Target size: {}x{}", config.targetWidth, config.targetHeight);

    u32 totalImages = 0;

    for (const auto& classEntry : std::filesystem::directory_iterator(directoryPath)) {
        if (!classEntry.is_directory()) continue;

        const std::string className = classEntry.path().filename().string();
        if (!classMap.contains(className)) {
            classMap[className] = nextClassId++;
            classNamesList.push_back(className);
        }
        u32 classId = classMap[className];

        for (const auto& imageEntry : std::filesystem::directory_iterator(classEntry.path())) {
            if (!imageEntry.is_regular_file()) continue;

            // Простая проверка расширений
            std::string ext = imageEntry.path().extension().string();
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") continue;

            int width, height, channels;
            // stbi_load автоматически выделит память через malloc
            u8* rawData = stbi_load(imageEntry.path().string().c_str(), &width, &height, &channels, config.forceChannels);

            if (!rawData) {
                Log::Logger().warning("Failed to load image: {}", imageEntry.path().string());
                continue;
            }

            // Если forceChannels задан, stbi_load вернет именно столько каналов
            int actualChannels = (config.forceChannels > 0) ? static_cast<int>(config.forceChannels) : channels;

            if (dataset.inputs.empty()) {
                dataset.imageChannels = actualChannels;
                Log::Logger().info("Set dataset channels to: {} based on first image/config.", actualChannels);
            } else if (static_cast<u32>(actualChannels) != dataset.imageChannels) {
                 Log::Logger().warning("Skipping image with different channel count: {} (expected {})", actualChannels, dataset.imageChannels);
                 stbi_image_free(rawData);
                 continue;
            }

            // 1. Изменяем размер (Resizing)
            std::vector<u8> resizedData = resizeImage(rawData, width, height, actualChannels, config.targetWidth, config.targetHeight);

            // Освобождаем оригинальный буфер, он больше не нужен
            stbi_image_free(rawData);

            // 2. Конвертируем в Eigen::VectorXf и нормализуем [0, 255] -> [0.0, 1.0]
            u64 vectorSize = config.targetWidth * config.targetHeight * actualChannels;
            Eigen::VectorXf imageVector(vectorSize);

            // Используем OpenMP для ускорения нормализации больших изображений
            #pragma omp parallel for
            for (i64 i = 0; i < static_cast<i64>(vectorSize); ++i) {
                imageVector(i) = static_cast<f32>(resizedData[i]) / 255.0f;
            }

            dataset.inputs.push_back(std::move(imageVector));

            // 3. Создаем метку
            Eigen::VectorXf labelVector = Eigen::VectorXf::Zero(classMap.size()); // Временный размер
            labelVector(classId) = 1.0f;
            dataset.outputs.push_back(std::move(labelVector));
            totalImages++;

            if (totalImages % 100 == 0) {
                Log::Logger().debug("Processed {} images...", totalImages);
            }
        }
    }

    // Корректируем размер one-hot векторов, так как теперь мы знаем общее кол-во классов
    for (auto& outputVec : dataset.outputs) {
        outputVec.conservativeResize(nextClassId);
    }
    dataset.classNames = std::move(classNamesList);

    if (dataset.inputs.empty()) {
        return std::unexpected(Error{"No valid images found in the directory."});
    }

    Log::Logger().info("Successfully loaded and resized {} images from {} classes.", totalImages, classMap.size());
    for(const auto& name : dataset.classNames) {
        Log::Logger().debug("Class found: {}", name);
    }

    return dataset;
}