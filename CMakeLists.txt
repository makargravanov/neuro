cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_MODULE_STD 1)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)

project(neuro LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
        boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG        boost-1.89.0
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
)

FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
)

set(EIGEN_BUILD_DOCS OFF CACHE BOOL "Disable Eigen docs" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen testing" FORCE)
set(EIGEN_BUILD_EXAMPLES OFF CACHE BOOL "Disable Eigen examples" FORCE)

FetchContent_MakeAvailable(boost eigen)

add_compile_options(-fmodules-ts)

file(GLOB_RECURSE MODULES_DIR "src/*.cppm")

add_executable(main main.cpp)
target_sources(main PUBLIC
        FILE_SET myModules
        TYPE CXX_MODULES
        BASE_DIRS src
        FILES ${MODULES_DIR}
)

target_include_directories(main PRIVATE ${boost_SOURCE_DIR})
target_include_directories(main PRIVATE ${eigen_SOURCE_DIR})

target_compile_options(main PRIVATE -fmodules-ts)

target_link_libraries(main PRIVATE stdc++exp)