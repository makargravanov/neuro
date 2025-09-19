//
// Created by Alex on 19.09.2025.
//

#ifndef OCLBUFFER_HPP
#define OCLBUFFER_HPP

#include "OclCore.hpp"
#include "../../../types/eigen_types.hpp"

class OclBuffer {
    cl::Buffer _buffer;
    size_t _size_bytes;

public:
    OclBuffer() : _size_bytes(0) {}

    OclBuffer(OclBuffer&& other) noexcept : _buffer(std::move(other._buffer)), _size_bytes(other._size_bytes) {
        other._size_bytes = 0;
    }

    OclBuffer& operator=(OclBuffer&& other) noexcept {
        if (this != &other) {
            _buffer = std::move(other._buffer);
            _size_bytes = other._size_bytes;
            other._size_bytes = 0;
        }
        return *this;
    }

    OclBuffer(const OclBuffer&) = delete;
    OclBuffer& operator=(const OclBuffer&) = delete;


    void create(size_t size) {
        if (size == 0) return;
        _size_bytes = size;
        _buffer = cl::Buffer(OclCore::getInstance().getContext(), CL_MEM_READ_WRITE, _size_bytes);
    }

    void write(const void* host_data) {
        if (_size_bytes > 0)
            OclCore::getInstance().getQueue().enqueueWriteBuffer(_buffer, CL_TRUE, 0, _size_bytes, host_data);
    }

    void read(void* host_data) const {
        if (_size_bytes > 0)
            OclCore::getInstance().getQueue().enqueueReadBuffer(_buffer, CL_TRUE, 0, _size_bytes, host_data);
    }

    const cl::Buffer& get() const { return _buffer; }
    cl::Buffer& get() { return _buffer; }
    [[nodiscard]] size_t size() const { return _size_bytes; }
};
#endif //OCLBUFFER_HPP
