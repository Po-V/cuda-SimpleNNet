#include <cuda_runtime.h>
#include <stdexcept>
#include<iostream>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

static inline void check_cuda_error(cudaError_t result, const char* func, const char* file, int line){
    if(result){
        std::cerr << "CUDA error at " <<file << ':' << line << "code =" << static_cast<unsigned int>(result)
        << "(" << cudaGetErrorString(result) << ") \"" << func << "\"" << std::endl;
        throw std::runtime_error("CUDA error");
    }
}