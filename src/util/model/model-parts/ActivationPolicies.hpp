
#ifndef ACTIVATION_POLICIES_HPP
#define ACTIVATION_POLICIES_HPP

#include <cmath>

#include "../../types/eigen_types.hpp"

enum class PolicyType {
    SIGMOID,
    LINEAR,
    RELU
    SOFTMAX // Добавляем новый тип
};


struct SigmoidPolicy {
    static f32 activate(f32 x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    static f32 derivative(f32 activatedX) {
        return activatedX * (1.0f - activatedX);
    }
};

struct LinearPolicy {
    static f32 activate(f32 x) {
        return x;
    }
    static f32 derivative(f32 activatedX) {
        return 1.0f;
    }
};

struct ReLUPolicy {
    static f32 activate(f32 x) {
        return std::max(0.0f, x);
    }
    static f32 derivative(f32 activatedX) {
        return activatedX > 0.0f ? 1.0f : 0.0f;
    }
};


struct SoftmaxPolicy {
    static f32 activate(f32 x) {
        return x;
    }

    static f32 derivative(f32 activatedX) {
        return 1.0f;
    }
};

#endif