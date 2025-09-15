cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_MODULE_STD 1)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)

project(neuro LANGUAGES CXX)

include(FetchContent)

set(BOOST_ENABLE_CMAKE ON)

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
)

FetchContent_Declare(
        nlohmann-json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.12.0
)

FetchContent_MakeAvailable(boost eigen nlohmann-json)

add_compile_options(-fmodules-ts)

file(GLOB_RECURSE HPP_DIR "src/*.hpp")
file(GLOB_RECURSE CPP_DIR "src/*.cpp")
file(GLOB_RECURSE MODULES_DIR "src/*.cppm")

add_executable(main
        main.cpp
        ${HPP_DIR}
        ${CPP_DIR}
)
target_sources(main PUBLIC
        FILE_SET myModules
        TYPE CXX_MODULES
        BASE_DIRS src
        FILES ${MODULES_DIR}
)

target_compile_options(main PRIVATE -fmodules-ts)

target_link_libraries(main PRIVATE
        stdc++exp
        Boost::system
        Boost::filesystem
        Boost::thread
        Boost::asio
        Boost::beast
        eigen
        nlohmann-json
)

if(WIN32)
    target_link_libraries(main PRIVATE ws2_32 mswsock)
endif()