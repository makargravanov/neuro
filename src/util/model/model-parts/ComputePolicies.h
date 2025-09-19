//
// Created by Alex on 19.09.2025.
//

#ifndef COMPUTEPOLICIES_H
#define COMPUTEPOLICIES_H


#include "../../types/eigen_types.hpp"
#include "ActivationPolicies.hpp"
#include "ocl-core/OclBuffer.hpp"
#include "ocl-core/OclCore.hpp"

/**
 * @struct CpuEigenPolicy
 * @brief Политика вычислений, использующая библиотеку Eigen для операций на CPU.
 *
 * Инкапсулирует все математические операции, необходимые для прямого и обратного
 * распространения сигнала в нейронной сети.
 */
struct CpuEigenPolicy {
    /**
     * @brief Выполняет основной шаг прямого распространения: z = (W * X).colwise() + b.
     */
    static Input forwardPass(const WeightMatrix& weights, const Input& input, const BiasVector& biases) {
        return (weights * input).colwise() + biases;
    }

    /**
     * @brief Применяет функцию активации к выходу линейного слоя.
     * @tparam ActivationPolicy Политика функции активации (например, SigmoidPolicy, SoftmaxPolicy).
     * @param z Входные данные для функции активации (результат forwardPass).
     * @return Результат после применения функции активации.
     */
    template<typename ActivationPolicy>
    static Output activate(const Input& z) {
        if constexpr (std::is_same_v<ActivationPolicy, SoftmaxPolicy>) {
            Eigen::RowVectorXf maxCoeffs = z.colwise().maxCoeff();
            Output expZ = (z.rowwise() - maxCoeffs).array().exp();
            return expZ.array().rowwise() / expZ.colwise().sum().array();
        } else {
            return z.unaryExpr(&ActivationPolicy::activate);
        }
    }

    /**
     * @brief Вычисляет производную функции активации.
     */
    template<typename ActivationPolicy>
    static Output activationDerivative(const Output& lastOutput) {
        return lastOutput.unaryExpr(&ActivationPolicy::derivative);
    }

    /**
     * @brief Вычисляет градиент для матрицы весов.
     */
    static WeightMatrix calculateWeightGradient(const Output& delta, const Input& prevLayerOutput) {
        return delta * prevLayerOutput.transpose();
    }

    /**
     * @brief Вычисляет градиент для вектора смещений.
     */
    static BiasVector calculateBiasGradient(const Output& delta) {
        return delta.rowwise().mean();
    }

    /**
     * @brief Обновляет веса слоя с использованием градиентного спуска.
     */
    static void updateWeights(WeightMatrix& weights, f32 learningRate, const WeightMatrix& weightGrad) {
        weights -= learningRate * weightGrad;
    }

    /**
     * @brief Обновляет смещения слоя с использованием градиентного спуска.
     */
    static void updateBiases(BiasVector& biases, f32 learningRate, const BiasVector& biasGrad) {
        biases -= learningRate * biasGrad;
    }

    /**
     * @brief Вычисляет ошибку (delta) для передачи на предыдущий слой.
     */
    static Output calculateNextDelta(const WeightMatrix& currentWeights, const Output& delta, const Output& prevActivationDerivative) {
        return (currentWeights.transpose() * delta).cwiseProduct(prevActivationDerivative);
    }

    /**
     * @brief Выполняет поэлементное умножение двух матриц.
     */
    static Output elementwiseProduct(const Output& a, const Output& b) {
        return a.cwiseProduct(b);
    }
};

/**
 * @struct GpuPolicy
 * @brief Заглушка для будущей реализации политики вычислений на GPU.
 *
 * Здесь могли бы быть реализованы те же статические методы, что и в CpuEigenPolicy,
 * но с использованием CUDA, OpenCL или других GPU-ускоренных библиотек.
 */

struct GpuPolicy {
public:
    static void init() {
        static bool initialized = false;
        if (!initialized) {
            OclCore::getInstance().compileKernel("kernels.cl");
            initialized = true;
        }
    }

    // --- Прямое распространение ---
    static OclBuffer forwardPass(const OclBuffer& weights, const OclBuffer& input, const OclBuffer& biases, u32 M, u32 K, u32 N) {
        // Эта функция - заглушка, т.к. мы используем fused ядра
        // В реальности здесь было бы matrix_multiply + add_bias
        throw std::logic_error("Use fused forward_pass kernels instead");
    }

    template<typename ActivationPolicy>
    static OclBuffer activate(const OclBuffer& input, const OclBuffer& weights, const OclBuffer& biases, u32 M, u32 K, u32 N) {
        OclBuffer output_buf;
        output_buf.create(M * N * sizeof(f32));
        auto& ocl = OclCore::getInstance();

        const char* kernel_name = "";
        if constexpr (std::is_same_v<ActivationPolicy, ReLUPolicy>) {
            kernel_name = "forward_pass_relu";
        } else if constexpr (std::is_same_v<ActivationPolicy, LinearPolicy>) {
            kernel_name = "forward_pass_linear";
        } else if constexpr (std::is_same_v<ActivationPolicy, SigmoidPolicy>) {
            kernel_name = "forward_pass_sigmoid";
        } else if constexpr (std::is_same_v<ActivationPolicy, SoftmaxPolicy>) {
            // Softmax в два шага
            OclBuffer col_sums_buf;
            col_sums_buf.create(N * sizeof(f32)); // N = batch_size

            cl::Kernel kernel1(ocl.getProgram(), "forward_pass_softmax_part1");
            kernel1.setArg(0, weights.get());
            kernel1.setArg(1, input.get());
            kernel1.setArg(2, biases.get());
            kernel1.setArg(3, output_buf.get()); // Здесь пока exp(z)
            kernel1.setArg(4, col_sums_buf.get());
            kernel1.setArg(5, static_cast<i32>(M));
            kernel1.setArg(6, static_cast<i32>(N));
            kernel1.setArg(7, static_cast<i32>(K));
            ocl.getQueue().enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(N), cl::NullRange);

            cl::Kernel kernel2(ocl.getProgram(), "forward_pass_softmax_part2");
            kernel2.setArg(0, output_buf.get());
            kernel2.setArg(1, col_sums_buf.get());
            kernel2.setArg(2, static_cast<i32>(M));
            kernel2.setArg(3, static_cast<i32>(N));
            ocl.getQueue().enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(M, N), cl::NullRange);

            return output_buf;

        } else {
            throw std::runtime_error("Unsupported activation policy for GPU");
        }

        cl::Kernel kernel(ocl.getProgram(), kernel_name);
        kernel.setArg(0, weights.get());
        kernel.setArg(1, input.get());
        kernel.setArg(2, biases.get());
        kernel.setArg(3, output_buf.get());
        kernel.setArg(4, static_cast<i32>(M));
        kernel.setArg(5, static_cast<i32>(N));
        kernel.setArg(6, static_cast<i32>(K));
        ocl.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M, N), cl::NullRange);
        return output_buf;
    }


    template<typename ActivationPolicy>
    static OclBuffer activationDerivative(const OclBuffer& lastOutput, u32 M, u32 N) {
        OclBuffer derivative_buf;
        derivative_buf.create(M * N * sizeof(f32));
        auto& ocl = OclCore::getInstance();
        const char* kernel_name = "";

        if constexpr (std::is_same_v<ActivationPolicy, ReLUPolicy>) {
            kernel_name = "relu_derivative";
        } else if constexpr (std::is_same_v<ActivationPolicy, SigmoidPolicy>) {
            kernel_name = "sigmoid_derivative";
        } else {
            // Fallback для Linear, Softmax (где производная сложная или равна 1)
            Output lastOutput_cpu(M, N);
            lastOutput.read(lastOutput_cpu.data());
            Output derivative_cpu = CpuEigenPolicy::activationDerivative<ActivationPolicy>(lastOutput_cpu);
            derivative_buf.write(derivative_cpu.data());
            return derivative_buf;
        }

        cl::Kernel kernel(ocl.getProgram(), kernel_name);
        kernel.setArg(0, lastOutput.get());
        kernel.setArg(1, derivative_buf.get());
        ocl.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M * N), cl::NullRange);
        return derivative_buf;
    }

    static OclBuffer calculateWeightGradient(const OclBuffer& delta, const OclBuffer& prevLayerOutput, u32 M, u32 N, u32 K) {
        auto& ocl = OclCore::getInstance();

        // grad = delta * prevLayerOutput^T
        // delta: M x N, prevLayerOutput: K x N => prevLayerOutput^T: N x K
        // grad: M x K

        OclBuffer prevLayerOutput_T;
        prevLayerOutput_T.create(N * K * sizeof(f32));
        cl::Kernel transpose_kernel(ocl.getProgram(), "matrix_transpose");
        transpose_kernel.setArg(0, prevLayerOutput.get());
        transpose_kernel.setArg(1, prevLayerOutput_T.get());
        transpose_kernel.setArg(2, static_cast<i32>(K));
        transpose_kernel.setArg(3, static_cast<i32>(N));
        ocl.getQueue().enqueueNDRangeKernel(transpose_kernel, cl::NullRange, cl::NDRange(K, N), cl::NullRange);

        OclBuffer gradient_buf;
        gradient_buf.create(M * K * sizeof(f32));
        cl::Kernel mm_kernel(ocl.getProgram(), "matrix_multiply");
        mm_kernel.setArg(0, delta.get());
        mm_kernel.setArg(1, prevLayerOutput_T.get());
        mm_kernel.setArg(2, gradient_buf.get());
        mm_kernel.setArg(3, static_cast<i32>(M));
        mm_kernel.setArg(4, static_cast<i32>(K));
        mm_kernel.setArg(5, static_cast<i32>(N));
        ocl.getQueue().enqueueNDRangeKernel(mm_kernel, cl::NullRange, cl::NDRange(M, K), cl::NullRange);

        return gradient_buf;
    }

    static OclBuffer calculateBiasGradient(const OclBuffer& delta, u32 M, u32 N) {
        OclBuffer bias_grad_buf;
        bias_grad_buf.create(M * sizeof(f32));
        auto& ocl = OclCore::getInstance();
        cl::Kernel kernel(ocl.getProgram(), "calculate_bias_gradient");
        kernel.setArg(0, delta.get());
        kernel.setArg(1, bias_grad_buf.get());
        kernel.setArg(2, static_cast<i32>(M));
        kernel.setArg(3, static_cast<i32>(N));
        ocl.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M), cl::NullRange);
        return bias_grad_buf;
    }

    static void updateWeights(OclBuffer& weights, f32 learningRate, const OclBuffer& weightGrad) {
        auto& ocl = OclCore::getInstance();
        cl::Kernel kernel(ocl.getProgram(), "update_weights_sgd");
        kernel.setArg(0, weights.get());
        kernel.setArg(1, weightGrad.get());
        kernel.setArg(2, learningRate);
        ocl.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(weights.size() / sizeof(f32)), cl::NullRange);
    }

    static void updateBiases(OclBuffer& biases, f32 learningRate, const OclBuffer& biasGrad) {
        updateWeights(biases, learningRate, biasGrad); // То же ядро подходит
    }

    static OclBuffer calculateNextDelta(const OclBuffer& currentWeights, const OclBuffer& delta, const OclBuffer& prevActivationDerivative, u32 M, u32 K, u32 N) {
        auto& ocl = OclCore::getInstance();

        // next_delta = (currentWeights^T * delta) .* prevActivationDerivative
        // currentWeights: K x M => currentWeights^T: M x K
        // delta: M x N
        // result of matmul: K x N

        OclBuffer weights_T;
        weights_T.create(M * K * sizeof(f32));
        cl::Kernel transpose_kernel(ocl.getProgram(), "matrix_transpose");
        transpose_kernel.setArg(0, currentWeights.get());
        transpose_kernel.setArg(1, weights_T.get());
        transpose_kernel.setArg(2, static_cast<i32>(K));
        transpose_kernel.setArg(3, static_cast<i32>(M));
        ocl.getQueue().enqueueNDRangeKernel(transpose_kernel, cl::NullRange, cl::NDRange(K, M), cl::NullRange);

        OclBuffer temp_delta;
        temp_delta.create(K * N * sizeof(f32));
        cl::Kernel mm_kernel(ocl.getProgram(), "matrix_multiply");
        mm_kernel.setArg(0, weights_T.get());
        mm_kernel.setArg(1, delta.get());
        mm_kernel.setArg(2, temp_delta.get());
        mm_kernel.setArg(3, static_cast<i32>(K));
        mm_kernel.setArg(4, static_cast<i32>(N));
        mm_kernel.setArg(5, static_cast<i32>(M));
        ocl.getQueue().enqueueNDRangeKernel(mm_kernel, cl::NullRange, cl::NDRange(K, N), cl::NullRange);

        OclBuffer next_delta_buf;
        next_delta_buf.create(K * N * sizeof(f32));
        cl::Kernel ep_kernel(ocl.getProgram(), "elementwise_product");
        ep_kernel.setArg(0, temp_delta.get());
        ep_kernel.setArg(1, prevActivationDerivative.get());
        ep_kernel.setArg(2, next_delta_buf.get());
        ocl.getQueue().enqueueNDRangeKernel(ep_kernel, cl::NullRange, cl::NDRange(K*N), cl::NullRange);

        return next_delta_buf;
    }

    static OclBuffer elementwiseProduct(const OclBuffer& a, const OclBuffer& b, u32 size) {
        OclBuffer result_buf;
        result_buf.create(size * sizeof(f32));
        auto& ocl = OclCore::getInstance();
        cl::Kernel kernel(ocl.getProgram(), "elementwise_product");
        kernel.setArg(0, a.get());
        kernel.setArg(1, b.get());
        kernel.setArg(2, result_buf.get());
        ocl.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
        return result_buf;
    }
};

#endif //COMPUTEPOLICIES_H
