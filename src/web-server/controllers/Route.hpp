//
// Created by Alex on 17.09.2025.
//

#ifndef ROUTE_H
#define ROUTE_H

#include <string>
#include <vector>
#include <set>
#include <regex>
#include <stdexcept>
#include <boost/beast/http/verb.hpp>

class Route {
    std::string _pathTemplate;            // Оригинальный путь, например "/datasets/{id}"
    std::regex _pathRegex;                // Скомпилированный regex, например "/datasets/([^/]+)"
    std::vector<std::string> _paramNames; // Имена параметров, например {"id"}

public:
    const std::set<http::verb> methods;

    explicit Route(std::string pathTemplate, std::set<http::verb> methods)
        : _pathTemplate(pathTemplate), methods(std::move(methods)) {

        std::string regexString = "^" + pathTemplate + "$";
        std::regex paramRegex("\\{([^}]+)\\}");
        auto wordsBegin = std::sregex_iterator(pathTemplate.begin(), pathTemplate.end(), paramRegex);
        auto wordsEnd = std::sregex_iterator();

        for (std::sregex_iterator i = wordsBegin; i != wordsEnd; ++i) {
            _paramNames.push_back((*i)[1].str());
        }

        regexString = std::regex_replace(regexString, paramRegex, "([^/]+)");
        _pathRegex = std::regex(regexString);
    }
    [[nodiscard]] const std::string& getPathTemplate() const { return _pathTemplate; }
    [[nodiscard]] const std::regex& getPathRegex() const { return _pathRegex; }
    [[nodiscard]] const std::vector<std::string>& getParamNames() const { return _paramNames; }

    bool operator<(const Route& other) const {
        return _pathTemplate < other._pathTemplate;
    }
};

#endif