//
// Created by Alex on 15.09.2025.
//

#ifndef EIGEN_TYPES_HPP
#define EIGEN_TYPES_HPP

#include <variant>
#include <Eigen/Dense>
#include "types.hpp"
#include "unsupported/Eigen/CXX11/src/Tensor/Tensor.h"


// --- Типы для полносвязных (Dense) слоев ---
// (batch_size, num_features)
using DenseInput = Eigen::MatrixXf;
using DenseOutput = Eigen::MatrixXf;
// Матрица весов для слоя
using WeightMatrix = Eigen::MatrixXf;
// Вектор смещений для слоя
using BiasVector = Eigen::VectorXf;


// --- Типы для сверточных (Convolutional) слоев ---
// Тензоры будут иметь формат NCHW (Batch, Channels, Height, Width)
// Это стандартный и наиболее производительный формат для многих фреймворков.
using Tensor4f = Eigen::Tensor<f32, 4, Eigen::RowMajor, Eigen::DenseIndex>;

// Веса для сверточного слоя
// (output_channels, input_channels, kernel_height, kernel_width)
using KernelTensor = Eigen::Tensor<f32, 4, Eigen::RowMajor, Eigen::DenseIndex>;


// --- Универсальные псевдонимы для использования в Network.hpp ---
// std::variant поможет нам обрабатывать разные форматы данных между слоями.
using InputType = std::variant<DenseInput, Tensor4f>;
using OutputType = std::variant<DenseOutput, Tensor4f>;


// Старые псевдонимы для обратной совместимости с существующим кодом,
// который мы будем рефакторить постепенно.
using Input = DenseInput;
using Output = DenseOutput;



#endif //EIGEN_TYPES_HPP