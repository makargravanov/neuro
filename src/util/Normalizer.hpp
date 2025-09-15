
#ifndef NORMALIZER_HPP
#define NORMALIZER_HPP

#include "types.hpp"
#include "eigen_types.hpp"
#include <vector>
#include <limits>


class Normalizer {
    f32 _min = std::numeric_limits<f32>::max();
    f32 _max = std::numeric_limits<f32>::lowest();

public:
    void fit(const std::vector<Input>& data, u32 columnIndex) {
        for (const auto& row : data) {
            if (row(columnIndex) < _min) _min = row(columnIndex);
            if (row(columnIndex) > _max) _max = row(columnIndex);
        }
    }

    [[nodiscard]] f32 transform(f32 value) const {
        if (_max - _min == 0) return 0;
        return (value - _min) / (_max - _min);
    }

    [[nodiscard]] f32 inverseTransform(f32 value) const {
        return value * (_max - _min) + _min;
    }
};

#endif
