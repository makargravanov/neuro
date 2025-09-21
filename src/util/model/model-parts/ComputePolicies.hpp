// --- START OF FILE ComputePolicies.h ---

#ifndef COMPUTEPOLICIES_H
#define COMPUTEPOLICIES_H

#include <unsupported/Eigen/CXX11/Tensor>
#include "../../types/eigen_types.hpp"
#include "ActivationPolicies.hpp"
#include "layers/ConvLayer.hpp"

/**
 * @struct CpuEigenPolicy
 * @brief Политика вычислений, использующая библиотеку Eigen для операций на CPU.
 *
 * Инкапсулирует все математические операции, необходимые для прямого и обратного
 * распространения сигнала в нейронной сети, включая операции для полносвязных
 * и сверточных слоев.
 */
struct CpuEigenPolicy {
    enum class ConvolutionMode { VALID, SAME, FULL };
    /**
     * @brief Универсальная вспомогательная функция свертки.
     * @details Инкапсулирует логику паддинга для разных режимов.
     */
    static Tensor4f _convolution(const Tensor4f& input, const KernelTensor& kernel, u32 stride, ConvolutionMode mode) {
        Tensor4f paddedInput = input;
        const Eigen::DenseIndex inHeight = input.dimension(2);
        const Eigen::DenseIndex inWidth = input.dimension(3);
        const Eigen::DenseIndex kernelHeight = kernel.dimension(2);
        const Eigen::DenseIndex kernelWidth = kernel.dimension(3);

        if (mode == ConvolutionMode::FULL) {
            const Eigen::DenseIndex padH = kernelHeight - 1;
            const Eigen::DenseIndex padW = kernelWidth - 1;
            paddedInput = input.pad(std::array<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>, 4>{
                std::pair{0, 0}, std::pair{0, 0}, std::pair{padH, padH}, std::pair{padW, padW}
            });
        } else if (mode == ConvolutionMode::SAME) {
            const Eigen::DenseIndex outHeight = (inHeight + stride - 1) / stride;
            const Eigen::DenseIndex outWidth = (inWidth + stride - 1) / stride;
            const Eigen::DenseIndex padAlongHeight = std::max((outHeight - 1) * stride + kernelHeight - inHeight, 0LL);
            const Eigen::DenseIndex padAlongWidth = std::max((outWidth - 1) * stride + kernelWidth - inWidth, 0LL);
            const Eigen::DenseIndex padTop = padAlongHeight / 2;
            const Eigen::DenseIndex padBottom = padAlongHeight - padTop;
            const Eigen::DenseIndex padLeft = padAlongWidth / 2;
            const Eigen::DenseIndex padRight = padAlongWidth - padLeft;
            paddedInput = input.pad(std::array<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>, 4>{
                std::pair{0, 0}, std::pair{0, 0}, std::pair{padTop, padBottom}, std::pair{padLeft, padRight}
            });
        }
        // для VALID используется исходный input

        Eigen::array<Eigen::DenseIndex, 2> strides = {static_cast<Eigen::DenseIndex>(stride), static_cast<Eigen::DenseIndex>(stride)};
        return paddedInput.convolve(kernel, strides);
    }

    // ===================================================================================
    // MARK: - 1. Общие утилиты и операции активации
    // ===================================================================================

    /**
     * @brief Применяет функцию активации к выходу линейного слоя (версия для Matrix).
     */
    template<typename ActivationPolicy>
    static DenseOutput activate(const DenseInput& z) {
        if constexpr (std::is_same_v<ActivationPolicy, SoftmaxPolicy>) {
            Eigen::RowVectorXf maxCoeffs = z.colwise().maxCoeff();
            DenseOutput expZ = (z.rowwise() - maxCoeffs).array().exp();
            return expZ.array().rowwise() / expZ.colwise().sum().array();
        } else {
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
    static Tensor4f convolution(const Tensor4f& input, const KernelTensor& kernels, const BiasVector& biases, u32 stride, PaddingMode paddingMode) {
        ConvolutionMode mode = (paddingMode == PaddingMode::SAME) ? ConvolutionMode::SAME : ConvolutionMode::VALID;
        Tensor4f output = _convolution(input, kernels, stride, mode);

        const Eigen::DenseIndex batchSize = output.dimension(0);
        const Eigen::DenseIndex outChannels = output.dimension(1);
        const Eigen::DenseIndex outHeight = output.dimension(2);
        const Eigen::DenseIndex outWidth = output.dimension(3);

        Tensor4f biasTensor(1, outChannels, 1, 1);
        for(Eigen::DenseIndex i = 0; i < outChannels; ++i) {
            biasTensor(0, i, 0, 0) = biases(i);
        }
        Eigen::array<Eigen::DenseIndex, 4> bcast = {batchSize, 1, outHeight, outWidth};
        return output + biasTensor.broadcast(bcast);
    }

    /**
     * @brief Обратное распространение: вычисление градиента для весов (ядер).
     * @details grad_W = conv(Input, Delta_output). Мы используем трюк с перестановкой
     *          измерений, чтобы свести задачу к стандартной свертке Eigen.
     */
    static KernelTensor calculateKernelGradient(const Tensor4f& prevLayerOutput, const Tensor4f& delta, u32 stride, PaddingMode paddingMode) {
        Tensor4f shuffledInput = prevLayerOutput.shuffle(std::array<int, 4>{1, 0, 2, 3});
        Tensor4f shuffledDelta = delta.shuffle(std::array<int, 4>{1, 0, 2, 3});

        // Градиент по весам - это свертка входа с дельтой.
        // Режим паддинга здесь противоположен прямому проходу.
        ConvolutionMode mode = (paddingMode == PaddingMode::SAME) ? ConvolutionMode::VALID : ConvolutionMode::SAME;

        // Для вычисления градиента мы используем дилатацию (dilation) входа, чтобы эмулировать stride
        Tensor4f dilatedInput = shuffledInput;
        if (stride > 1) {
            const auto& ddims = shuffledInput.dimensions();
            Eigen::DSizes<Eigen::DenseIndex, 4> upsampled_dims(
                ddims[0], ddims[1],
                (ddims[2] - 1) * stride + 1,
                (ddims[3] - 1) * stride + 1
            );
            dilatedInput.resize(upsampled_dims);
            dilatedInput.setZero();
            Eigen::array<Eigen::DenseIndex, 4> offsets = {0, 0, 0, 0};
            Eigen::array<Eigen::DenseIndex, 4> extents = {ddims[0], ddims[1], ddims[2], ddims[3]};
            Eigen::array<Eigen::DenseIndex, 4> strides_arr = {1, 1, static_cast<Eigen::DenseIndex>(stride), static_cast<Eigen::DenseIndex>(stride)};
            dilatedInput.stridedSlice(offsets, extents, strides_arr) = shuffledInput;
        }

        Tensor4f grad = _convolution(dilatedInput, shuffledDelta, 1, mode);

        return grad.shuffle(std::array<int, 4>{1, 0, 2, 3});
    }
    static BiasVector calculateBiasGradient(const Tensor4f& delta) {
        Eigen::array<Eigen::DenseIndex, 3> sum_dims = {0, 2, 3};
        Eigen::Tensor<f32, 1, Eigen::RowMajor> biasGradTensor = delta.sum(sum_dims);
        return Eigen::Map<BiasVector>(biasGradTensor.data(), biasGradTensor.dimension(0));
    }

    /**
     * @brief Обратное распространение: вычисление дельты для предыдущего слоя.
     * @details delta_prev = full_conv(Delta_output, rot180(Kernel)). Это реализуется
     *          как транспонированная свертка.
     */
    static Tensor4f calculateNextDeltaConv(const KernelTensor& kernels, const Tensor4f& delta, const Tensor4f& prevActivationDerivative, u32 stride, PaddingMode paddingMode) {
        auto rotatedKernels = kernels.reverse(std::array<bool, 4>{false, false, true, true});
        auto shuffledKernels = rotatedKernels.shuffle(std::array<int, 4>{1, 0, 2, 3});

        Tensor4f nextDeltaZ = _convolution(delta, shuffledKernels, 1, ConvolutionMode::FULL);

        if (stride > 1) {
            const auto& shape = nextDeltaZ.dimensions();
            Eigen::array<Eigen::DenseIndex, 4> start_indices = {0, 0, 0, 0};
            Eigen::array<Eigen::DenseIndex, 4> stop_indices = {shape[0], shape[1], shape[2], shape[3]};
            Eigen::array<Eigen::DenseIndex, 4> strides_arr = {1, 1, static_cast<Eigen::DenseIndex>(stride), static_cast<Eigen::DenseIndex>(stride)};
            nextDeltaZ = nextDeltaZ.stridedSlice(start_indices, stop_indices, strides_arr);
        }

        const auto& prevShape = prevActivationDerivative.dimensions();
        Eigen::array<Eigen::DenseIndex, 4> start_indices = {0, 0, 0, 0};
        Eigen::array<Eigen::DenseIndex, 4> slice_sizes = {prevShape[0], prevShape[1], prevShape[2], prevShape[3]};
        nextDeltaZ = nextDeltaZ.slice(start_indices, slice_sizes);

        return nextDeltaZ * prevActivationDerivative;
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

                        Eigen::Tensor<f32, 0, Eigen::RowMajor> maxValTensor = window.maximum();
                        output(n, c, i, j) = maxValTensor();

                        Eigen::Tensor<Eigen::DenseIndex, 0, Eigen::RowMajor> maxIndexTensor = window.argmax();
                        const Eigen::DenseIndex maxIndex_local = maxIndexTensor();

                        // Сохраняем глобальный индекс в исходном тензоре
                        Eigen::DenseIndex row = maxIndex_local / poolSize;
                        Eigen::DenseIndex col = maxIndex_local % poolSize;
                        indices(n, c, i, j) = (h_start + row) * inWidth + (w_start + col);
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