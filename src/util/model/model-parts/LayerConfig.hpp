//
// Created by Alex on 21.09.2025.
//

#ifndef LAYERCONFIG_HPP
#define LAYERCONFIG_HPP


#include <vector>
#include <variant>
#include "layers/ConvLayer.hpp" // For PaddingMode
#include "ActivationPolicies.hpp" // For LayerType
#include "../../types/types.hpp"

// Конфигурация для сверточного слоя
struct ConvLayerConfig {
    u32 outputChannels;
    u32 kernelSize;
    u32 stride = 1;
    PaddingMode paddingMode = PaddingMode::VALID;
    // Активация задается через LayerType
};

// Конфигурация для слоя пулинга
struct PoolLayerConfig {
    u32 poolSize;
    u32 stride;
};

// Конфигурация для полносвязного слоя
struct DenseLayerConfig {
    u32 numberOfNeurons;
    // Активация задается через LayerType
};

// variant для хранения любой возможной конфигурации
using AnyLayerConfig = std::variant<
    ConvLayerConfig,
    PoolLayerConfig,
    DenseLayerConfig
>;

// Структура для связи типа слоя с его конфигурацией
struct LayerConfig {
    LayerType type;
    AnyLayerConfig config;
};

// Псевдоним для вектора конфигураций, описывающего всю архитектуру
using ArchitectureConfig = std::vector<LayerConfig>;


#endif //LAYERCONFIG_HPP
