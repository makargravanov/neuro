// --- START OF FILE ComputePolicies.h ---

#ifndef COMPUTEPOLICIES_H
#define COMPUTEPOLICIES_H

#include <unsupported/Eigen/CXX11/Tensor>
#include "../../types/eigen_types.hpp"
#include "ActivationPolicies.hpp"
#include "layers/PoolingLayer.hpp"

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

    template<typename T, int Rank, int Options>
    static std::string getTensorDims(const Eigen::Tensor<T, Rank, Options>& tensor) {
        std::string dims = "[";
        for (int i = 0; i < Rank; ++i) {
            dims += std::to_string(tensor.dimension(i));
            if (i < Rank - 1) {
                dims += ", ";
            }
        }
        dims += "]";
        return dims;
    }

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

        // --- ИСПРАВЛЕНИЕ НАЧИНАЕТСЯ ЗДЕСЬ ---

        // 1. Преобразуем тензоры в формат, ожидаемый Eigen (NHWC и HWIO)
        // NCHW -> NHWC
        Eigen::array<int, 4> nchw_to_nhwc = {0, 2, 3, 1};
        auto input_nhwc = paddedInput.shuffle(nchw_to_nhwc);

        // [C_out, C_in, kH, kW] -> [kH, kW, C_in, C_out]
        Eigen::array<int, 4> oihw_to_hwio = {2, 3, 1, 0};
        auto kernel_hwio = kernel.shuffle(oihw_to_hwio);

        // 2. Выполняем свертку
        Eigen::array<Eigen::DenseIndex, 2> strides = {static_cast<Eigen::DenseIndex>(stride), static_cast<Eigen::DenseIndex>(stride)};

        // Log::Logger().debug("Convolving NHWC input dims {} with HWIO kernel dims {}",
        //                   getTensorDims(input_nhwc), getTensorDims(kernel_hwio));

        auto output_nhwc = input_nhwc.convolve(kernel_hwio, strides);

        // 3. Преобразуем результат обратно в NCHW
        // NHWC -> NCHW
        Eigen::array<int, 4> nhwc_to_nchw = {0, 3, 1, 2};
        return output_nhwc.shuffle(nhwc_to_nchw);
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
        // --- Финальный лог для диагностики ---
        Log::Logger().debug(
            "calculateNextDelta: Weights[sum:{}, min:{}, max:{}], Delta[sum:{}]",
            currentWeights.sum(),
            currentWeights.minCoeff(),
            currentWeights.maxCoeff(),
            delta.sum()
        );
        auto nextDelta = (currentWeights.transpose() * delta).cwiseProduct(prevActivationDerivative);
        Log::Logger().debug("calculateNextDelta: Resulting nextDelta sum = {}", nextDelta.sum());
        // --- Конец лога ---

        return nextDelta;
    }

    // ===================================================================================
    // MARK: - 3. Операции для сверточных слоев (Convolutional Layers)
    // ===================================================================================

    /**
     * @brief Прямое распространение: операция свертки.
     */
    static Tensor4f convolution(const Tensor4f& input, const KernelTensor& kernels, const BiasVector& biases, u32 stride, PaddingMode paddingMode) {
        const Eigen::DenseIndex batchSize = input.dimension(0);
        const Eigen::DenseIndex inChannels = input.dimension(1);
        const Eigen::DenseIndex inHeight = input.dimension(2);
        const Eigen::DenseIndex inWidth = input.dimension(3);

        const Eigen::DenseIndex outChannels = kernels.dimension(0);
        const Eigen::DenseIndex kernelHeight = kernels.dimension(2);
        const Eigen::DenseIndex kernelWidth = kernels.dimension(3);

        // TODO: Реализовать паддинг для режима SAME
        if (paddingMode == PaddingMode::SAME) {
            throw std::runtime_error("PaddingMode::SAME is not implemented for loop-based convolution yet.");
        }

        const Eigen::DenseIndex outHeight = (inHeight - kernelHeight) / stride + 1;
        const Eigen::DenseIndex outWidth = (inWidth - kernelWidth) / stride + 1;

        Tensor4f output(batchSize, outChannels, outHeight, outWidth);
        output.setZero();

        // N (batch)
        for (Eigen::DenseIndex n = 0; n < batchSize; ++n) {
            // OC (output channels / filters)
            for (Eigen::DenseIndex oc = 0; oc < outChannels; ++oc) {
                // H_out
                for (Eigen::DenseIndex i = 0; i < outHeight; ++i) {
                    // W_out
                    for (Eigen::DenseIndex j = 0; j < outWidth; ++j) {

                        f32 sum = 0.0f;
                        // Вычисляем свертку для одного пикселя (i, j) на выходной карте признаков
                        // IC (input channels)
                        for (Eigen::DenseIndex ic = 0; ic < inChannels; ++ic) {
                            // kH (kernel height)
                            for (Eigen::DenseIndex kh = 0; kh < kernelHeight; ++kh) {
                                // kW (kernel width)
                                for (Eigen::DenseIndex kw = 0; kw < kernelWidth; ++kw) {
                                    const Eigen::DenseIndex h_in = i * stride + kh;
                                    const Eigen::DenseIndex w_in = j * stride + kw;
                                    sum += input(n, ic, h_in, w_in) * kernels(oc, ic, kh, kw);
                                }
                            }
                        }
                        output(n, oc, i, j) = sum + biases(oc);
                    }
                }
            }
        }
        return output;
    }

    /**
     * @brief Обратное распространение: вычисление градиента для весов (ядер).
     * @details grad_W = conv(Input, Delta_output). Мы используем трюк с перестановкой
     *          измерений, чтобы свести задачу к стандартной свертке Eigen.
     */
    static KernelTensor calculateKernelGradient(const Tensor4f& prevLayerOutput, const Tensor4f& delta, u32 stride, PaddingMode paddingMode) {
        const Eigen::DenseIndex batchSize = prevLayerOutput.dimension(0);
        const Eigen::DenseIndex inChannels = prevLayerOutput.dimension(1);
        const Eigen::DenseIndex inHeight = prevLayerOutput.dimension(2);
        const Eigen::DenseIndex inWidth = prevLayerOutput.dimension(3);

        const Eigen::DenseIndex outChannels = delta.dimension(1);
        const Eigen::DenseIndex outHeight = delta.dimension(2);
        const Eigen::DenseIndex outWidth = delta.dimension(3);

        const Eigen::DenseIndex kernelHeight = inHeight - (outHeight - 1) * stride;
        const Eigen::DenseIndex kernelWidth = inWidth - (outWidth - 1) * stride;

        KernelTensor kernelGrad(outChannels, inChannels, kernelHeight, kernelWidth);
        kernelGrad.setZero();

        for (Eigen::DenseIndex n = 0; n < batchSize; ++n) {
            for (Eigen::DenseIndex oc = 0; oc < outChannels; ++oc) {
                for (Eigen::DenseIndex ic = 0; ic < inChannels; ++ic) {
                    for (Eigen::DenseIndex kh = 0; kh < kernelHeight; ++kh) {
                        for (Eigen::DenseIndex kw = 0; kw < kernelWidth; ++kw) {
                            f32 sum = 0.0f;
                            for (Eigen::DenseIndex i = 0; i < outHeight; ++i) {
                                for (Eigen::DenseIndex j = 0; j < outWidth; ++j) {
                                    const Eigen::DenseIndex h_in = i * stride + kh;
                                    const Eigen::DenseIndex w_in = j * stride + kw;
                                    sum += prevLayerOutput(n, ic, h_in, w_in) * delta(n, oc, i, j);
                                }
                            }
                            kernelGrad(oc, ic, kh, kw) += sum;
                        }
                    }
                }
            }
        }
        return kernelGrad;
    }

    static BiasVector calculateBiasGradient(const Tensor4f& delta) {
        Eigen::array<Eigen::DenseIndex, 3> sum_dims = {0, 2, 3};
        Eigen::Tensor<f32, 1, Eigen::RowMajor> biasGradTensor = delta.sum(sum_dims);
        return Eigen::Map<BiasVector>(biasGradTensor.data(), biasGradTensor.dimension(0));
    }

    /**
     * @brief Обратное распространение: вычисление дельты для предыдущего слоя. (Надежная реализация на циклах - транспонированная свертка)
     */
    static Tensor4f calculateNextDeltaConv(const KernelTensor& kernels, const Tensor4f& delta, const Tensor4f& prevActivationDerivative, u32 stride, PaddingMode paddingMode) {
        const Eigen::DenseIndex batchSize = delta.dimension(0);
        const Eigen::DenseIndex inChannels = kernels.dimension(1);
        const Eigen::DenseIndex outChannels = kernels.dimension(0);
        const Eigen::DenseIndex kernelHeight = kernels.dimension(2);
        const Eigen::DenseIndex kernelWidth = kernels.dimension(3);

        const Eigen::DenseIndex outHeight = delta.dimension(2);
        const Eigen::DenseIndex outWidth = delta.dimension(3);

        const Eigen::DenseIndex inHeight = prevActivationDerivative.dimension(2);
        const Eigen::DenseIndex inWidth = prevActivationDerivative.dimension(3);

        Tensor4f nextDeltaZ(batchSize, inChannels, inHeight, inWidth);
        nextDeltaZ.setZero();

        for (Eigen::DenseIndex n = 0; n < batchSize; ++n) {
            for (Eigen::DenseIndex ic = 0; ic < inChannels; ++ic) {
                for (Eigen::DenseIndex i = 0; i < outHeight; ++i) {
                    for (Eigen::DenseIndex j = 0; j < outWidth; ++j) {
                        for (Eigen::DenseIndex oc = 0; oc < outChannels; ++oc) {
                            for (Eigen::DenseIndex kh = 0; kh < kernelHeight; ++kh) {
                                for (Eigen::DenseIndex kw = 0; kw < kernelWidth; ++kw) {
                                    const Eigen::DenseIndex h_in = i * stride + kh;
                                    const Eigen::DenseIndex w_in = j * stride + kw;
                                    if (h_in < inHeight && w_in < inWidth) {
                                        nextDeltaZ(n, ic, h_in, w_in) += delta(n, oc, i, j) * kernels(oc, ic, kh, kw);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

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
    template<typename CP>
    static std::pair<Tensor4f, typename PoolingLayer<CP>::MaxIndicesTensor>
    maxPooling(const Tensor4f& input, u32 poolSize, u32 stride)  {
        // --- ЛОГ ДЛЯ ПРЯМОГО ПРОХОДА ПУЛИНГА ---
        Eigen::Tensor<f32, 0, Eigen::RowMajor> inputSum = input.sum();
        Log::Logger().debug("MaxPool IN: input_dims={}, input_sum={}, pool_size={}, stride={}",
                            getTensorDims(input), static_cast<f32>(inputSum()), poolSize, stride);
        // --- КОНЕЦ ЛОГА ---

        const Eigen::DenseIndex batchSize = input.dimension(0);
        const Eigen::DenseIndex channels = input.dimension(1);
        const Eigen::DenseIndex inHeight = input.dimension(2);
        const Eigen::DenseIndex inWidth = input.dimension(3);

        const Eigen::DenseIndex outHeight = (inHeight - poolSize) / stride + 1;
        const Eigen::DenseIndex outWidth = (inWidth - poolSize) / stride + 1;

        Tensor4f output(batchSize, channels, outHeight, outWidth);
        typename PoolingLayer<CP>::MaxIndicesTensor indices(batchSize, channels, outHeight, outWidth);

        for (Eigen::DenseIndex n = 0; n < batchSize; ++n) {
            for (Eigen::DenseIndex c = 0; c < channels; ++c) {
                for (Eigen::DenseIndex i = 0; i < outHeight; ++i) {
                    for (Eigen::DenseIndex j = 0; j < outWidth; ++j) {
                        const Eigen::DenseIndex hStart = i * stride;
                        const Eigen::DenseIndex wStart = j * stride;

                        // Извлекаем окно/патч
                        auto window = input.slice(
                            std::array<Eigen::DenseIndex, 4>{n, c, hStart, wStart},
                            std::array<Eigen::DenseIndex, 4>{1, 1, static_cast<i64>(poolSize), static_cast<i64>(poolSize)}
                        );

                        // Находим максимум и его индекс внутри этого окна
                        Eigen::Tensor<f32, 0, Eigen::RowMajor> maxValTensor = window.maximum();
                        output(n, c, i, j) = maxValTensor();

                        Eigen::Tensor<Eigen::DenseIndex, 0, Eigen::RowMajor> maxIndexTensor = window.argmax();
                        const Eigen::DenseIndex flatIndexInWindow = maxIndexTensor();

                        // Преобразуем "плоский" индекс в окне в 2D-координаты в окне
                        const Eigen::DenseIndex rowInWindow = flatIndexInWindow / poolSize;
                        const Eigen::DenseIndex colInWindow = flatIndexInWindow % poolSize;

                        // Преобразуем координаты в окне в глобальные координаты в исходном тензоре
                        const Eigen::DenseIndex globalRow = hStart + rowInWindow;
                        const Eigen::DenseIndex globalCol = wStart + colInWindow;

                        // Сохраняем "сплющенный" глобальный индекс (относительно 2D-среза HxW)
                        indices(n, c, i, j) = globalRow * inWidth + globalCol;
                    }
                }
            }
        }

        // --- ЛОГ ДЛЯ ВЫХОДА ПРЯМОГО ПРОХОДА ПУЛИНГА ---
        Eigen::Tensor<f32, 0, Eigen::RowMajor> outputSum = output.sum();
        Eigen::Tensor<Eigen::DenseIndex, 0, Eigen::RowMajor> indicesSum = indices.sum();
        Log::Logger().debug("MaxPool OUT: output_dims={}, output_sum={}, indices_sum={}",
                            getTensorDims(output), static_cast<f32>(outputSum()), static_cast<Eigen::DenseIndex>(indicesSum()));
        // --- КОНЕЦ ЛОГА ---

        return std::make_pair(output, indices);
    }

    /**
     * @brief Обратное распространение для Max Pooling. (Надежная реализация на циклах)
     */
    static Tensor4f maxPoolingBackward(const Tensor4f& delta, const typename PoolingLayer<CpuEigenPolicy>::MaxIndicesTensor& maxIndices, const std::array<Eigen::DenseIndex, 4>& prevShape) {
        // --- ЛОГ ДЛЯ ОБРАТНОГО ПРОХОДА ПУЛИНГА ---
        Eigen::Tensor<f32, 0, Eigen::RowMajor> deltaSum = delta.sum();
        Eigen::Tensor<Eigen::DenseIndex, 0, Eigen::RowMajor> indicesSum = maxIndices.sum();
        Log::Logger().debug("MaxPoolBack IN: delta_dims={}, delta_sum={}, indices_sum={}, prev_shape=[{},{},{},{}]",
                            getTensorDims(delta), static_cast<f32>(deltaSum()), static_cast<Eigen::DenseIndex>(indicesSum()),
                            prevShape[0], prevShape[1], prevShape[2], prevShape[3]);
        // --- КОНЕЦ ЛОГА ---

        Tensor4f prevDelta(prevShape[0], prevShape[1], prevShape[2], prevShape[3]);
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
                        // Получаем "сплющенный" глобальный индекс, который мы сохранили
                        Eigen::DenseIndex flatGlobalIndex = maxIndices(n, c, i, j);

                        // Преобразуем его обратно в 2D-координаты
                        Eigen::DenseIndex globalRow = flatGlobalIndex / inWidth;
                        Eigen::DenseIndex globalCol = flatGlobalIndex % inWidth;

                        // Передаем градиент в ту ячейку, откуда пришел максимум
                        if (globalRow < prevShape[2] && globalCol < prevShape[3]) {
                            prevDelta(n, c, globalRow, globalCol) += delta(n, c, i, j);
                        }
                    }
                }
            }
        }

        // --- ЛОГ ДЛЯ ВЫХОДА ОБРАТНОГО ПРОХОДА ПУЛИНГА ---
        Eigen::Tensor<f32, 0, Eigen::RowMajor> prevDeltaSum = prevDelta.sum();
        Log::Logger().debug("MaxPoolBack OUT: prev_delta_dims={}, prev_delta_sum={}",
                            getTensorDims(prevDelta), static_cast<f32>(prevDeltaSum()));
        // --- КОНЕЦ ЛОГА ---

        return prevDelta;
    }



    // ===================================================================================
    // MARK: - 5. Операции для слоя выравнивания (Flatten Layer)
    // ===================================================================================

    /**
     * @brief Преобразует 4D тензор в 2D матрицу.
     */
    static DenseOutput flatten(const Tensor4f& input) {
        // --- ЛОГ ДЛЯ ДИАГНОСТИКИ FLATTEN ---
        Eigen::Tensor<f32, 0, Eigen::RowMajor> inputSumTensor = input.sum();
        Log::Logger().debug("Flatten IN: Tensor sum = {}", static_cast<f32>(inputSumTensor()));

        // --- НОВЫЙ ЛОГ: ВЫВОД СОДЕРЖИМОГО ТЕНЗОРА ---
        if (input.dimension(0) < 5) { // Выводим только для небольших батчей
            std::stringstream ssTensor;
            for (Eigen::DenseIndex n = 0; n < input.dimension(0); ++n) {
                ssTensor << "Sample [" << n << "]:\n";
                for (Eigen::DenseIndex c = 0; c < input.dimension(1); ++c) {
                    ssTensor << "  Channel [" << c << "]:\n";
                    ssTensor << input.chip(n, 0).chip(c, 0) << "\n";
                }
            }
            Log::Logger().debug("Flatten IN Tensor content:\n{}", ssTensor.str());
        }
        // --- КОНЕЦ НОВОГО ЛОГА ---

        const Eigen::DenseIndex batchSize = input.dimension(0);
        const Eigen::DenseIndex features = input.dimension(1) * input.dimension(2) * input.dimension(3);

        DenseOutput result = Eigen::Map<const Eigen::MatrixXf>(input.data(), features, batchSize);

        Log::Logger().debug(
            "Flatten OUT: Matrix dims=[{}, {}], sum={}, min={}, max={}",
            result.rows(),
            result.cols(),
            result.sum(),
            result.minCoeff(),
            result.maxCoeff()
        );
        if (result.cols() < 5) {
            std::stringstream ssMatrix;
            ssMatrix << result;
            Log::Logger().debug("Flatten OUT Matrix content:\n{}", ssMatrix.str());
        }
        // --- КОНЕЦ ЛОГА FLATTEN ---

        return result;
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