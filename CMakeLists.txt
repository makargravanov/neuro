cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_MODULE_STD 1)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)

project(lesson1 LANGUAGES CXX)

add_compile_options(-fmodules-ts)

file(GLOB_RECURSE MODULES_DIR "src/*.cppm")

add_executable(main main.cpp)
target_sources(main PUBLIC
        FILE_SET myModules
        TYPE CXX_MODULES
        BASE_DIRS src
        FILES ${MODULES_DIR}
)

target_compile_options(main PRIVATE -fmodules-ts)

target_link_libraries(main PRIVATE stdc++exp)