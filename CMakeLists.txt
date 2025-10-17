cmake_minimum_required(VERSION 4.0)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_MODULE_STD 1)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)

project(neuro LANGUAGES CXX)

set(OPENCL_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencl-sdk)
set(OpenCL_INCLUDE_DIR ${OPENCL_SDK_DIR}/include)
set(OpenCL_LIBRARY ${OPENCL_SDK_DIR}/lib/OpenCL.lib)

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
        GIT_TAG        e0a59e5a66e6d16fa93ab4f5e48bf539205e837f
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
)

FetchContent_Declare(
        nlohmann-json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.12.0
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
)

FetchContent_MakeAvailable(boost eigen nlohmann-json stb)

add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${stb_SOURCE_DIR})

add_compile_options(-fmodules-ts)

file(GLOB_RECURSE HPP_DIR "src/*.hpp")
file(GLOB_RECURSE CPP_DIR "src/*.cpp")
file(GLOB_RECURSE MODULES_DIR "src/*.cppm")
file(GLOB_RECURSE KERNELS_DIR "src/util/model/model-parts/ocl_kernels/*.cl")

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

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

if(OpenCL_FOUND)

target_link_libraries(main PRIVATE
        OpenMP::OpenMP_CXX
        stdc++exp
        Boost::system
        Boost::thread
        Boost::asio
        Boost::beast
        Boost::uuid
        Eigen3::Eigen
        nlohmann_json
        OpenCL::OpenCL
        stb_image
)
endif()

if(WIN32)
    target_link_libraries(main PRIVATE ws2_32 mswsock)
endif()