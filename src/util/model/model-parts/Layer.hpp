
#ifndef LAYER_HPP
#define LAYER_HPP

#include "../../types/eigen_types.hpp"

template<typename ActivationPolicy>
class Layer {
    WeightMatrix _weights;
    BiasVector _biases;

    Input _lastInput;
    Output _lastOutput;
public:
    Layer() = default;

    explicit Layer(const u32 numberOfNeurons, u32 lastNumberOfNeurons) {
        _weights = WeightMatrix::Random(numberOfNeurons, lastNumberOfNeurons);
        _biases = BiasVector::Random(numberOfNeurons);
    };

    Output activate(const Input& input) {
        _lastInput = input;
        // z = W * X + b, где X - батч входов (каждый столбец - пример)
        // .colwise() добавляет вектор смещений к каждому столбцу матрицы
        Input z = (_weights * input).colwise() + _biases;

        if constexpr (std::is_same_v<ActivationPolicy, SoftmaxPolicy>) {
            // Применяем Softmax к каждому столбцу (примеру) в батче
            // Вычитаем максимум в каждом столбце для численной стабильности
            Eigen::RowVectorXf maxCoeffs = z.colwise().maxCoeff();
            Output expZ = (z.rowwise() - maxCoeffs).array().exp();
            // Нормализуем каждый столбец на его сумму
            _lastOutput = expZ.array().rowwise() / expZ.colwise().sum().array();
        } else {
            // Поэлементная активация для остальных политик
            _lastOutput = z.unaryExpr(&ActivationPolicy::activate);
        }

        return _lastOutput;
    }


    WeightMatrix& getWeights() { return _weights; }
    BiasVector& getBiases() { return _biases; }
    [[nodiscard]] const WeightMatrix& getWeights() const { return _weights; }
    [[nodiscard]] const BiasVector& getBiases() const { return _biases; }

    [[nodiscard]] const Output& getLastOutput() const { return _lastOutput; }
    [[nodiscard]] const Input& getLastInput() const { return _lastInput; }

    [[nodiscard]] Output activationDerivative() const {
        return _lastOutput.unaryExpr(&ActivationPolicy::derivative);
    }
};


#endif
