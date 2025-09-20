// --- START OF FILE ComputePolicies.h ---

#ifndef COMPUTEPOLICIES_H
#define COMPUTEPOLICIES_H

#include <unsupported/Eigen/CXX11/Tensor>
#include "../../types/eigen_types.hpp"
#include "ActivationPolicies.hpp"

/**
 * @struct CpuEigenPolicy
 * @brief Политика вычислений, использующая библиотеку Eigen для операций на CPU.
 *
 * Инкапсулирует все математические операции, необходимые для прямого и обратного
 * распространения сигнала в нейронной сети, включая операции для полносвязных
 * и сверточных слоев.
 */
struct CpuEigenPolicy {

    // ===================================================================================
    // MARK: - 1. Общие утилиты и операции активации
    // ===================================================================================

    /**
     * @brief Применяет функцию активации к выходу линейного слоя (версия для Matrix).
     */
    template<typename ActivationPolicy>
    static DenseOutput activate(const DenseInput& z) {
        if constexpr (std::is_same_v<ActivationPolicy, SoftmaxPolicy>) {
            // Softmax не является поэлементной, требует особого подхода
            Eigen::RowVectorXf maxCoeffs = z.colwise().maxCoeff();
            DenseOutput expZ = (z.rowwise() - maxCoeffs).array().exp();
            return expZ.array().rowwise() / expZ.colwise().sum().array();
        } else {
            // Для поэлементных функций (ReLU, Sigmoid, Linear)
            return z.unaryExpr(&ActivationPolicy::activate);
        }
    }

    /**
     * @brief Применяет функцию активации к выходу сверточного слоя (версия для Tensor).
     */
    template<typename ActivationPolicy>
    static Tensor4f activate(const Tensor4f& z) {
        // Все активации в сверточных слоях (ReLU, Linear) являются поэлементными.
        return z.unaryExpr(&ActivationPolicy::activate);
    }


    /**
     * @brief Вычисляет производную функции активации (версия для Matrix).
     */
    template<typename ActivationPolicy>
    static DenseOutput activationDerivative(const DenseOutput& lastOutput) {
        return lastOutput.unaryExpr(&ActivationPolicy::derivative);
    }

    /**
     * @brief Вычисляет производную функции активации (версия для Tensor).
     */
    template<typename ActivationPolicy>
    static Tensor4f activationDerivative(const Tensor4f& lastOutput) {
        return lastOutput.unaryExpr(&ActivationPolicy::derivative);
    }

    /**
     * @brief Выполняет поэлементное умножение двух матриц.
     */
    static DenseOutput elementwiseProduct(const DenseOutput& a, const DenseOutput& b) {
        return a.cwiseProduct(b);
    }

    /**
     * @brief Выполняет поэлементное умножение двух тензоров.
     */
    static Tensor4f elementwiseProduct(const Tensor4f& a, const Tensor4f& b) {
        return a * b;
    }

    // ===================================================================================
    // MARK: - 2. Операции для полносвязных слоев (Dense Layers)
    // ===================================================================================

    static DenseInput forwardPass(const WeightMatrix& weights, const DenseInput& input, const BiasVector& biases) {
        return (weights * input).colwise() + biases;
    }

    static WeightMatrix calculateWeightGradient(const DenseOutput& delta, const DenseInput& prevLayerOutput) {
        return delta * prevLayerOutput.transpose();
    }

    static BiasVector calculateBiasGradient(const DenseOutput& delta) {
        return delta.rowwise().mean();
    }

    static void updateWeights(WeightMatrix& weights, f32 learningRate, const WeightMatrix& weightGrad) {
        weights -= learningRate * weightGrad;
    }

    static void updateBiases(BiasVector& biases, f32 learningRate, const BiasVector& biasGrad) {
        biases -= learningRate * biasGrad;
    }

    static DenseOutput calculateNextDelta(const WeightMatrix& currentWeights, const DenseOutput& delta, const DenseOutput& prevActivationDerivative) {
        return (currentWeights.transpose() * delta).cwiseProduct(prevActivationDerivative);
    }

    // ===================================================================================
    // MARK: - 3. Операции для сверточных слоев (Convolutional Layers)
    // ===================================================================================

    /**
     * @brief Прямое распространение: операция свертки.
     */
    static Tensor4f convolution(const Tensor4f& input, const KernelTensor& kernels, const BiasVector& biases, u32 stride, u32 padding) {
        const Eigen::DenseIndex batchSize = input.dimension(0);
        const Eigen::DenseIndex outChannels = kernels.dimension(0);
        const Eigen::DenseIndex inHeight = input.dimension(2);
        const Eigen::DenseIndex inWidth = input.dimension(3);
        const Eigen::DenseIndex kernelHeight = kernels.dimension(2);
        const Eigen::DenseIndex kernelWidth = kernels.dimension(3);

        const Eigen::DenseIndex outHeight = (inHeight - kernelHeight + 2 * padding) / stride + 1;
        const Eigen::DenseIndex outWidth = (inWidth - kernelWidth + 2 * padding) / stride + 1;

        Tensor4f paddedInput = input;
        if (padding > 0) {
            paddedInput = input.pad(std::array<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>, 4>{
                std::pair{0, 0}, std::pair{0, 0}, std::pair{padding, padding}, std::pair{padding, padding}
            }, 0.0f);
        }

        // Используем встроенную свертку в Eigen::Tensor
        Eigen::array<Eigen::DenseIndex, 2> strides = {stride, stride};
        Tensor4f output = paddedInput.convolve(kernels, strides);

        // Добавляем смещения (biases)
        Tensor4f biasTensor(1, outChannels, 1, 1);
        for(Eigen::DenseIndex i = 0; i < outChannels; ++i) {
            biasTensor(0, i, 0, 0) = biases(i);
        }
        // Используем broadcasting для добавления смещения к каждому элементу в канале
        Eigen::array<Eigen::DenseIndex, 4> bcast = {batchSize, 1, outHeight, outWidth};
        Tensor4f broadcasted_biases = biasTensor.broadcast(bcast);
        output = output + broadcasted_biases;

        return output;
    }

    /**
     * @brief Обратное распространение: вычисление градиента для весов (ядер).
     */
    static KernelTensor calculateKernelGradient(const Tensor4f& delta, const Tensor4f& prevLayerOutput, u32 kernelH, u32 kernelW, u32 stride, u32 padding) {
        // Это свертка входа с дельтой выхода
        // TODO: Реализовать более сложную логику с учетом stride и padding
        return KernelTensor(); // Заглушка
    }

    /**
     * @brief Обратное распространение: вычисление градиента для смещений.
     */
    static BiasVector calculateBiasGradient(const Tensor4f& delta) {
        // Суммируем градиенты по всем измерениям, кроме каналов
        Eigen::array<Eigen::DenseIndex, 3> sum_dims = {0, 2, 3};
        Eigen::Tensor<f32, 1> biasGradTensor = delta.sum(sum_dims);
        return Eigen::Map<BiasVector>(biasGradTensor.data(), biasGradTensor.dimension(0));
    }

    /**
     * @brief Обратное распространение: вычисление дельты для предыдущего слоя.
     */
    static Tensor4f calculateNextDeltaConv(const KernelTensor& kernels, const Tensor4f& delta, const Tensor4f& prevActivationDerivative, u32 stride, u32 padding) {
        // Это "полная" свертка дельты с повернутыми на 180 градусов ядрами
        // TODO: Реализовать
        return Tensor4f(); // Заглушка
    }

    /**
     * @brief Обновление весов (ядер) для сверточного слоя.
     */
    static void updateWeights(KernelTensor& kernels, f32 learningRate, const KernelTensor& kernelGrad) {
            kernels = kernels - (kernelGrad * learningRate);
    }


    // ===================================================================================
    // MARK: - 4. Операции для слоев пулинга (Pooling Layers)
    // ===================================================================================

    /**
     * @brief Прямое распространение: Max Pooling.
     * @return Пара: результат пулинга и тензор с индексами максимальных элементов.
     */
    static std::pair<Tensor4f, Eigen::Tensor<Eigen::DenseIndex, 4>>
    maxPooling(const Tensor4f& input, u32 poolSize, u32 stride) {
        const Eigen::DenseIndex batchSize = input.dimension(0);
        const Eigen::DenseIndex channels = input.dimension(1);
        const Eigen::DenseIndex inHeight = input.dimension(2);
        const Eigen::DenseIndex inWidth = input.dimension(3);

        const Eigen::DenseIndex outHeight = (inHeight - poolSize) / stride + 1;
        const Eigen::DenseIndex outWidth = (inWidth - poolSize) / stride + 1;

        Tensor4f output(batchSize, channels, outHeight, outWidth);
        Eigen::Tensor<Eigen::DenseIndex, 4> indices(batchSize, channels, outHeight, outWidth);

        // TODO: Эту операцию можно значительно оптимизировать с помощью Eigen::patch и reductions,
        // но для наглядности приведена простая реализация.
        for (Eigen::DenseIndex n = 0; n < batchSize; ++n) {
            for (Eigen::DenseIndex c = 0; c < channels; ++c) {
                for (Eigen::DenseIndex i = 0; i < outHeight; ++i) {
                    for (Eigen::DenseIndex j = 0; j < outWidth; ++j) {
                        Eigen::DenseIndex h_start = i * stride;
                        Eigen::DenseIndex w_start = j * stride;

                        // Вырезаем "окно" из входного тензора
                        auto window = input.chip(n, 0).chip(c, 0)
                                          .slice(std::array<Eigen::DenseIndex, 2>{h_start, w_start},
                                                 std::array<Eigen::DenseIndex, 2>{poolSize, poolSize});

                        Eigen::Index maxIndex_local; // Создаем локальную переменную для сохранения индекса максимума

                        // ИСПРАВЛЕНО: Создаем 0-мерный тензор для хранения скалярного результата.
                        // Присваивание выражения window.maximum(maxIndex_local) этому 0-мерному тензору
                        // принудительно выполнит вычисление максимума и сохранит его значение,
                        // а также заполнит maxIndex_local.
                        Eigen::Tensor<f32, 0, Eigen::RowMajor, Eigen::DenseIndex> maxValScalar;
                        maxValScalar = window.maximum(maxIndex_local); // Вычисление происходит здесь

                        // Теперь maxValScalar содержит скалярное значение максимума.
                        // Извлекаем его с помощью оператора () для 0-мерных тензоров.
                        output(n, c, i, j) = maxValScalar();

                        // Сохраняем глобальный индекс в исходном тензоре, используя maxIndex_local
                        Eigen::Index row = maxIndex_local / poolSize;
                        Eigen::Index col = maxIndex_local % poolSize;
                        indices(n, c, i, j) = (h_start + row) * inWidth + (w_start + col);
                    }
                }
            }
        }
        return {output, indices};
    }

    /**
     * @brief Обратное распространение для Max Pooling.
     */
    static Tensor4f maxPoolingBackward(const Tensor4f& delta, const Eigen::Tensor<Eigen::DenseIndex, 4>& maxIndices, const std::array<Eigen::DenseIndex, 4>& prevShape) {
        Tensor4f prevDelta(prevShape);
        prevDelta.setZero();

        const Eigen::DenseIndex batchSize = delta.dimension(0);
        const Eigen::DenseIndex channels = delta.dimension(1);
        const Eigen::DenseIndex outHeight = delta.dimension(2);
        const Eigen::DenseIndex outWidth = delta.dimension(3);
        const Eigen::DenseIndex inWidth = prevShape[3];

        for (Eigen::DenseIndex n = 0; n < batchSize; ++n) {
            for (Eigen::DenseIndex c = 0; c < channels; ++c) {
                for (Eigen::DenseIndex i = 0; i < outHeight; ++i) {
                    for (Eigen::DenseIndex j = 0; j < outWidth; ++j) {
                        Eigen::DenseIndex globalIndex = maxIndices(n, c, i, j);
                        Eigen::DenseIndex h = globalIndex / inWidth;
                        Eigen::DenseIndex w = globalIndex % inWidth;
                        prevDelta(n, c, h, w) += delta(n, c, i, j);
                    }
                }
            }
        }
        return prevDelta;
    }


    // ===================================================================================
    // MARK: - 5. Операции для слоя выравнивания (Flatten Layer)
    // ===================================================================================

    /**
     * @brief Преобразует 4D тензор в 2D матрицу.
     */
    static DenseOutput flatten(const Tensor4f& input) {
        const Eigen::DenseIndex batchSize = input.dimension(0);
        const Eigen::DenseIndex features = input.dimension(1) * input.dimension(2) * input.dimension(3);
        // Важно: транспонируем, чтобы получить формат (features, batch_size)
        return Eigen::Map<const Eigen::MatrixXf>(input.data(), features, batchSize);
    }

    /**
     * @brief Обратное преобразование из 2D матрицы в 4D тензор.
     */
    static Tensor4f unflatten(const DenseInput& delta, const std::array<Eigen::DenseIndex, 4>& originalShape) {
        return Tensor4f(Eigen::TensorMap<const Tensor4f>(delta.data(), originalShape));
    }
};


/**
 * @struct GpuPolicy
 * @brief Заглушка для будущей реализации политики вычислений на GPU.
 */
struct GpuPolicy {
    // TODO: Реализовать вычисления на GPU с использованием CUDA/cuDNN
};

#endif //COMPUTEPOLICIES_H