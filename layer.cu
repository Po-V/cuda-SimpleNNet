#include "layer.cuh"
#include <cmath>
#include <iostream>
#include<curand_kernel.h>
#include <cuda_runtime.h>

__global__ void initialize_weights(float* weights_d, int size, unsigned long long seed){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        // store information to generate sequence of random numbers;
        curandState state;
        // seed represents random number sequence starting point, idx allows sequence to start at different points
        curand_init(seed + idx, idx, 0, &state); // state is a pointer to curandState structure that will be initialized
        weights_d[idx] = curand_normal(&state) *sqrtf(2.0f / size); // He initialization
    }
}

Layer::Layer(int input_size, int output_size, int activtion_type): input_size(input_size), output_size(output_size), 
             activation_type(activation_type){
    
    cudaMalloc(&input_d, input_size*sizeof(float));
    cudaMalloc(&output_d, output_size*sizeof(float));
    cudaMalloc(&weights_d, input_size*output_size*sizeof(float));
    cudaMalloc(&bias_d, input_size*output_size*sizeof(float));
    cudaMalloc(&input_error_d, input_size*sizeof(float));
    cudaMalloc(&output_error_d, output_size*sizeof(float));

    int numThreadsPerBlock = 128;
    int numBlocks = (input_size*output_size + numThreadsPerBlock - 1) / numThreadsPerBlock;

    printf("Launching kernel initialize_weights with %d blocks and %d threads per block\n", numBlocks, numThreadsPerBlock);
    initialize_weights<<<numBlocks, numThreadsPerBlock>>>(weights_d, input_size*output_size, time(nullptr));
    cudaDeviceSynchronize();

    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(launch_error));
        throw std::runtime_error("Kernel launch failed");
    }

    cudaMemset(bias_d, 0, output_size*sizeof(float));
}

Layer::~Layer(){
    cudaFree(weights_d);
    cudaFree(bias_d);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(input_error_d);
    cudaFree(output_error_d);
}

__global__ void forward_kernel(float* input, float*output, float* weights, float* bias, int input_size, int output_size, 
                               int activation_type){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < output_size){
        float output_val = bias[idx];
        for(int i = 0; i < input_size; ++i){
            output_val += input[i] * weights[idx*input_size+i];
        }
        if(activation_type ==0){
            output_val = fmaxf(0.0f, output_val);
        }else{
            output_val = 1.0f / (1.0f + exp(-output_val));
        }

        output[idx] = output_val;
    }
}

std::vector<float> Layer::forward(const std:: vector<float>&input){
    cudaMemcpy(input_d, input.data(), input_size*sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;
    forward_kernel<<<grid_size, block_size>>>(input_d, output_d, weights_d, bias_d, input_size, output_size, activation_type);
    cudaDeviceSynchronize();
    // CHECK_CUDA_ERROR(cudaGetLastError());

    std::vector<float> output(output_size);
    cudaMemcpy(output.data(), output_d, output_size*sizeof(float), cudaMemcpyDeviceToHost);
    return output;
}

__global__ void backward_kernel(float* input, float* output, float* weights, float* output_error, int input_size, 
                                int output_size, int activation_type, float learning_rate, float* bias, float* input_error){
    int idx = blockIdx.x * blockDim.x +threadIdx.x;

    if(idx < output_size){
        float derivative;
        if(activation_type == 0){
            derivative = (output[idx] >0.0f) ? 1.0f :0.0f;
        }else{
            derivative = output[idx]*(1.0 - output[idx]);
        }
        float delta = output_error[idx]*derivative;

        // if(next_layer_error != nullptr){
        //     delta =0.0f;
        //     for(int i = 0; i < input_size; ++i){
        //         delta += next_layer_error[i] * weights[i*output_size+idx];
        //     }
        //     delta *= derivative;
        // }

        for(int i = 0; i < input_size; i++){
            atomicAdd(&weights[idx * input_size+i], -learning_rate*delta*input[i]);
            atomicAdd(&input_error[i], delta * weights[idx * input_size + i]);
        }
        atomicAdd(&bias[idx], -learning_rate*delta);
    }
}

std::vector<float> Layer::backward(const std::vector<float>& output_error, float learning_rate) {
    cudaMemcpy(output_error_d, output_error.data(), output_error.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(input_error_d, 0, input_size * sizeof(float));
    
    int blockSize = 256;
    int numBlocks = (output_size + blockSize - 1) / blockSize;
    backward_kernel<<<numBlocks, blockSize>>>(input_d, output_d, weights_d, output_error_d, input_size, output_size, 
                                                activation_type, learning_rate, bias_d, input_error_d);
    cudaDeviceSynchronize();                                            
    // CHECK_CUDA_ERROR(cudaGetLastError());

    
    std::vector<float> input_error(input_size);
    cudaMemcpy(input_error.data(), input_error_d, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    return input_error;
}



