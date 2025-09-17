//
// Created by Alex on 17.09.2025.
//

#ifndef TRANSFORMATIONSERVICE_H
#define TRANSFORMATIONSERVICE_H

#include "DatasetService.hpp"
#include <memory>
#include <string>

class TransformationService {
public:
    /**
       * @brief Создает новый датасет на основе исходного, но без указанной колонки.
       * @param source Указатель на исходный, неизменяемый датасет.
       * @param columnName Имя колонки для удаления.
       * @return Указатель на новый, измененный датасет.
       * @throws std::runtime_error если колонка с таким именем не найдена.
       */
    std::shared_ptr<Dataset> removeColumn(std::shared_ptr<const Dataset> source, const std::string& columnName);
};



#endif //TRANSFORMATIONSERVICE_H
