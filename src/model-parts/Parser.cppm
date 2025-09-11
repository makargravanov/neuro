
export module Parser;

import std;
import types;
import Neuron;

export class Parser {
    std::vector<Input> _inputs;
    std::vector<Output> _outputs;
    std::vector<std::string> _header;
    std::map<std::string, u32> _classMap;
    u32 _nextClassId = 0;

public:
    // Конструктор принимает путь к файлу, список колонок для входа и колонку для выхода
    explicit Parser(const std::string& filepath, const std::vector<u32>& featureColumns, u32 targetColumn, bool hasHeader = true, char delimiter = ',') {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filepath);
        }

        std::string line;

        // Если есть заголовок, считываем его
        if (hasHeader && std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, delimiter)) {
                _header.push_back(cell);
            }
        }

        // Читаем данные
        while (std::getline(file, line)) {
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, delimiter)) {
                row.push_back(cell);
            }

            if (row.size() <= std::max(*std::ranges::max_element(featureColumns), targetColumn)) {
                // Пропускаем строки с некорректным количеством столбцов
                continue;
            }

            // Формируем вектор входов
            Input currentInput;
            currentInput.reserve(featureColumns.size());
            for (u32 col_idx : featureColumns) {
                currentInput.push_back(std::stof(row.at(col_idx)));
            }
            _inputs.push_back(currentInput);

            // Формируем вектор выходов
            const std::string& targetValue = row.at(targetColumn);
            try {
                // Если выход - число (задача регрессии)
                f32 numericOutput = std::stof(targetValue);
                _outputs.push_back({numericOutput});
            } catch (const std::invalid_argument&) {
                // Если выход - категория (задача классификации)
                if (!_classMap.contains(targetValue)) {
                    _classMap[targetValue] = _nextClassId++;
                }
                // One-hot encoding будет выполнен позже, когда все классы будут известны
                // Пока что просто сохраняем ID класса
                 _outputs.push_back({static_cast<f32>(_classMap[targetValue])});
            }
        }
        
        // Если это была классификация, преобразуем выходы в one-hot
        if (_nextClassId > 0) {
            for(auto& outputVec : _outputs) {
                u32 classId = static_cast<u32>(outputVec[0]);
                outputVec.assign(_nextClassId, 0.0f); // Заполняем нулями
                outputVec[classId] = 1.0f; // Ставим единицу в нужной позиции
            }
        }
    }

    const std::vector<Input>& getInputs() const { return _inputs; }
    const std::vector<Output>& getOutputs() const { return _outputs; }
    u32 getOutputSize() const { return _nextClassId == 0 ? 1 : _nextClassId; }
    u32 getInputSize() const { return _inputs.empty() ? 0 : _inputs.at(0).size(); }
};