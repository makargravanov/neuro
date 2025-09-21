//
// Created by Alex on 20.09.2025.
//

#ifndef FLATTENLAYER_HPP
#define FLATTENLAYER_HPP
#include "../../../types/eigen_types.hpp"
#include "../ComputePolicies.hpp"

template<typename ComputePolicy>
class FlattenLayer {
    // Сохраняем исходные размеры для обратного распространения
    std::array<Eigen::DenseIndex, 4> _lastInputShape;
    DenseOutput _lastOutput;
public:
    FlattenLayer() = default;

    /**
     * @brief "Выравнивает" 4D-тензор в 2D-матрицу.
     * @param input 4D-тензор (N, C, H, W).
     * @return 2D-матрица (N, C * H * W).
     */
    DenseOutput activate(const Tensor4f& input) {
        _lastInputShape = input.dimensions();
        // Делегируем операцию изменения формы политике вычислений
        return ComputePolicy::flatten(input);
    }

    [[nodiscard]] const auto& getLastInputShape() const { return _lastInputShape; }

    [[nodiscard]] DenseOutput activationDerivative() const {
        return DenseOutput::Ones(_lastOutput.rows(), _lastOutput.cols());
    }
};
#endif //FLATTENLAYER_HPP
