
#ifndef PARSER_HPP
#define PARSER_HPP

#include <fstream>
#include <map>
#include <vector>
#include <sstream>

#include "../util/eigen_types.hpp"

class Parser {
    std::vector<Input> _inputs;
    std::vector<Output> _outputs;
    std::vector<std::string> _header;
    std::map<std::string, u32> _classMap;
    u32 _nextClassId = 0;

public:
    explicit Parser(const std::string& filepath, const std::vector<u32>& featureColumns, u32 targetColumn, bool hasHeader = true, char delimiter = ',') {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filepath);
        }

        std::string line;

        if (hasHeader && std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, delimiter)) {
                _header.push_back(cell);
            }
        }

        while (std::getline(file, line)) {
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, delimiter)) {
                row.push_back(cell);
            }

            if (row.size() <= std::max(*std::ranges::max_element(featureColumns), targetColumn)) {
                continue;
            }

            Input currentInput(featureColumns.size());
            for (i64 i = 0; i < featureColumns.size(); ++i) {
                currentInput(i) = std::stof(row.at(featureColumns[i]));
            }
            _inputs.push_back(currentInput);

            // Формируем вектор выходов (теперь Eigen::VectorXf)
            const std::string& targetValue = row.at(targetColumn);
            try {
                f32 numericOutput = std::stof(targetValue);
                Output out(1);
                out(0) = numericOutput;
                _outputs.push_back(out);
            } catch (const std::invalid_argument&) {
                if (!_classMap.contains(targetValue)) {
                    _classMap[targetValue] = _nextClassId++;
                }
                Output out(1);
                out(0) = static_cast<f32>(_classMap[targetValue]);
                _outputs.push_back(out);
            }
        }

        if (_nextClassId > 0) {
            for(auto& outputVec : _outputs) {
                u32 classId = static_cast<u32>(outputVec(0));
                outputVec.resize(_nextClassId);
                outputVec.setZero();
                outputVec(classId) = 1.0f;
            }
        }
    }

    [[nodiscard]] const std::vector<Input>& getInputs() const { return _inputs; }
    [[nodiscard]] const std::vector<Output>& getOutputs() const { return _outputs; }
    [[nodiscard]] u32 getOutputSize() const { return _nextClassId == 0 ? 1 : _nextClassId; }
    [[nodiscard]] u32 getInputSize() const { return _inputs.empty() ? 0 : _inputs.at(0).size(); }
};

#endif