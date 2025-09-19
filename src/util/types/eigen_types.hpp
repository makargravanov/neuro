//
// Created by Alex on 15.09.2025.
//

#ifndef EIGEN_TYPES_HPP
#define EIGEN_TYPES_HPP

#include <Eigen/Dense>
#include "types.hpp"

using MatrixXfRowMajor = Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXfRowMajor = Eigen::Matrix<f32, Eigen::Dynamic, 1>;

using Input = MatrixXfRowMajor;
using Output = MatrixXfRowMajor;
//матрица весов для слоя
using WeightMatrix = MatrixXfRowMajor;
//вектор смещений для слоя
using BiasVector = VectorXfRowMajor;

#endif //EIGEN_TYPES_HPP