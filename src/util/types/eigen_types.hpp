//
// Created by Alex on 15.09.2025.
//

#ifndef EIGEN_TYPES_HPP
#define EIGEN_TYPES_HPP

#include <Eigen/Dense>
#include "types.hpp"

using Input = Eigen::VectorXf;
using Output = Eigen::VectorXf;
//матрица весов для слоя
using WeightMatrix = Eigen::MatrixXf;
//вектор смещений для слоя
using BiasVector = Eigen::VectorXf;

#endif //EIGEN_TYPES_HPP