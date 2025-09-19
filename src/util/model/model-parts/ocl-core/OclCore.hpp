//
// Created by Alex on 19.09.2025.
//

#ifndef OCLCORE_HPP
#define OCLCORE_HPP


#include <CL/opencl.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "../../../logging.hpp"

class OclCore {
    cl::Context _context;
    cl::CommandQueue _queue;
    cl::Program _program;

public:
    OclCore() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        cl::Platform platform = platforms.front();
        Log::Logger().info("Using OpenCL platform: {}", platform.getInfo<CL_PLATFORM_NAME>());

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices found.");
        }

        cl::Device device = devices.front();
        Log::Logger().info("Using OpenCL device: {}", device.getInfo<CL_DEVICE_NAME>());

        _context = cl::Context(device);
        _queue = cl::CommandQueue(_context, device);

    }

    void compileKernel(const std::string& kernelPath) {
        std::ifstream kernelFile(kernelPath);
        if (!kernelFile.is_open()) {
            throw std::runtime_error("Could not open kernel file: " + kernelPath);
        }
        std::string sourceCode(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

        _program = cl::Program(_context, sourceCode);
        if (_program.build("-cl-std=CL1.2") != CL_SUCCESS) {
            throw std::runtime_error("Error building: " + _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(_context.getInfo<CL_CONTEXT_DEVICES>()[0]));
        }
        Log::Logger().info("OpenCL kernel compiled successfully from {}", kernelPath);
    }

    cl::Context& getContext() { return _context; }
    cl::CommandQueue& getQueue() { return _queue; }
    cl::Program& getProgram() { return _program; }

    static OclCore& getInstance() {
        static OclCore instance;
        return instance;
    }
};

#endif //OCLCORE_HPP
