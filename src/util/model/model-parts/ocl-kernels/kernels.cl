
__kernel void matrix_multiply(
    const __global float* A,
    const __global float* B,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    const int globalRow = get_global_id(0); // Row of C
    const int globalCol = get_global_id(1); // Col of C

    if (globalRow >= M || globalCol >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        // A[row, k] * B[k, col]
        acc += A[globalRow * K + k] * B[k * N + globalCol];
    }
    C[globalRow * N + globalCol] = acc;
}

// Транспонирование матрицы A[M, K] -> B[K, M]
__kernel void matrix_transpose(
    const __global float* A,
    __global float* B,
    const int M,
    const int K)
{
    const int globalRow = get_global_id(0); // row of A
    const int globalCol = get_global_id(1); // col of A

    if (globalRow >= M || globalCol >= K) return;

    B[globalCol * M + globalRow] = A[globalRow * K + globalCol];
}


// --- Ядра для прямого распространения (Fused) ---

// Умножение, добавление смещения и ReLU активация за один проход
__kernel void forward_pass_relu(
    const __global float* weights,  // W (M x K)
    const __global float* input,    // X (K x N)
    const __global float* biases,   // B (M x 1)
    __global float* output,         // Y (M x N)
    const int M,
    const int N,
    const int K)
{
    const int globalRow = get_global_id(0); // Row of output
    const int globalCol = get_global_id(1); // Col of output

    if (globalRow >= M || globalCol >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += weights[globalRow * K + k] * input[k * N + globalCol];
    }

    float biased_val = acc + biases[globalRow];
    output[globalRow * N + globalCol] = fmax(0.0f, biased_val);
}

// То же самое, но для линейной активации
__kernel void forward_pass_linear(
    const __global float* weights,
    const __global float* input,
    const __global float* biases,
    __global float* output,
    const int M,
    const int N,
    const int K)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += weights[globalRow * K + k] * input[k * N + globalCol];
    }
    output[globalRow * N + globalCol] = acc + biases[globalRow];
}

__kernel void calculate_bias_gradient(
    const __global float* delta, // M x N
    __global float* bias_grad,   // M x 1
    const int M,
    const int N)
{
    const int row = get_global_id(0); // Neuron index
    if (row >= M) return;

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += delta[row * N + j];
    }
    bias_grad[row] = sum / N; // Среднее по батчу
}


// Обновление весов: weights -= learning_rate * gradient
__kernel void update_weights_sgd(
    __global float* weights,
    const __global float* gradient,
    const float learning_rate)
{
    const int i = get_global_id(0);
    weights[i] -= learning_rate * gradient[i];
}

__kernel void forward_pass_sigmoid(
    const __global float* weights,
    const __global float* input,
    const __global float* biases,
    __global float* output,
    const int M,
    const int N,
    const int K)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if (globalRow >= M || globalCol >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += weights[globalRow * K + k] * input[k * N + globalCol];
    }

    float z = acc + biases[globalRow];
    // Применяем Sigmoid
    output[globalRow * N + globalCol] = 1.0f / (1.0f + exp(-z));
}

// --- Производная Sigmoid ---
// derivative = activated_output * (1.0f - activated_output)
__kernel void sigmoid_derivative(
    const __global float* activated_output,
    __global float* derivative)
{
    const int i = get_global_id(0);
    float val = activated_output[i];
    derivative[i] = val * (1.0f - val);
}


// --- Fused ядро для Softmax (более сложное) ---
// Это ядро будет выполнять только линейную часть + экспоненту.
// Финальное деление на сумму будет вторым шагом.
__kernel void forward_pass_softmax_part1(
    const __global float* weights,
    const __global float* input,
    const __global float* biases,
    __global float* output_exp, // Сюда пишем exp(z)
    __global float* col_sums,   // Сюда пишем суммы по столбцам
    const int M, // num_neurons
    const int N, // batch_size
    const int K) // prev_num_neurons
{
    // Это ядро работает по столбцам (по каждому примеру в батче)
    const int j = get_global_id(0); // Индекс столбца (sample in batch)
    if (j >= N) return;

    // 1. Найти max(z) в текущем столбце для стабильности
    float max_z = -FLT_MAX;
    for (int i = 0; i < M; ++i) {
        float z = biases[i];
        for (int k = 0; k < K; ++k) {
            z += weights[i * K + k] * input[k * N + j];
        }
        if (z > max_z) {
            max_z = z;
        }
    }

    // 2. Посчитать exp(z - max_z) и их сумму
    float sum_exp = 0.0f;
    for (int i = 0; i < M; ++i) {
        float z = biases[i];
        for (int k = 0; k < K; ++k) {
            z += weights[i * K + k] * input[k * N + j];
        }
        float exp_val = exp(z - max_z);
        output_exp[i * N + j] = exp_val;
        sum_exp += exp_val;
    }
    col_sums[j] = sum_exp;
}

// Шаг 2 для Softmax: деление на сумму
__kernel void forward_pass_softmax_part2(
    __global float* output,         // M x N
    const __global float* col_sums, // N x 1
    const int M,
    const int N)
{
    const int i = get_global_id(0); // neuron index
    const int j = get_global_id(1); // sample index
    if (i >= M || j >= N) return;

    output[i * N + j] /= col_sums[j];
}