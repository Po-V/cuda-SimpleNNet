// header guard to prevent duplicate definitions
#pragma once

#include <vector>
#include "layer.cuh"
// Reprsents a class for neural networks
class NeuralNetwork {
public:
    // constructs neural network with given layer size
    NeuralNetwork(const std::vector<int>&layer_size);
    // cleans up resources when object is destroyed
    ~NeuralNetwork();

    // perform forward propagation with vector input and produce vector outputs
    std::vector<float> forward(const std::vector<float> &input);
    void train(const std::vector<std::vector<float>> &inputs,
               const std::vector<std::vector<float>> &targets,
               float learning_rate, int epochs);
private:
    //represents internal structure of NN and is kept private
    std::vector<Layer> layers;

};