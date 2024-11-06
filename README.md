# CUDA Neural Network
Neural network implementation with single hidden layer in C/C++ with CUDA

## Features

- Flexible architecture supporting any number of layers and neurons
- CUDA-accelerated computations (without CUDNN)
- Optimization using gradient descent

## Requirements

- NVIDIA GPU with CUDA support.
- CUDA Toolkit (tested with version 11.0, but should work with newer version)
- C++11 compatible compiler
- Make build system

## Project Structure

- `main.cu`: Main file demonstrating the usage of the neural network
- `neural_network.cuh/cu`: Neural network class definition and implementation
- `layer.cuh/cu`: Layer class definition and implementation
- `test_layer.cu`: Contains functions to test forward and backward propagation of layers.
- `test_neural_network.cu`: Contains functions to test the neural network.
- `Makefile`: Build configuration

## Building the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/Po-V/cuda-SimpleNNet.git
   cd cuda-SimpleNNet

2. Modify the Makefile if necessary, asjusting the `NVCC_FLAGS to match your GPU.

3. Build the project:

    ```bash
    make

3. Run the main program after compiling with:

    ```bash
    make run
