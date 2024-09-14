#include "NeuralNetwork.cuh"
#include <cassert>
#include <iostream>

void test_nn_forward(){

    std::vector <int> layer_size = {10, 50, 2};
    NeuralNetwork nn(layer_size);

    std::vector<float> input(10, 0.5f);

    std::vector<float> output = nn.forward(input);

    assert(output.size() == 2);

    std::cout << "Neural network forward pass test passed!" <<std::endl;
}

void test_nn_training(){

    // Layer sizes : 10 inputs, 50 hidden units with 2 output units
    std::vector <int> layer_size = {10,50,2};
    NeuralNetwork nn(layer_size);

    // create synthetic data (50 samples with 10 features each)
    std::vector<std::vector<float>> inputs(50, std::vector<float>(10, 0.5f));
    std::vector<std::vector<float>> targets(50, std::vector<float>{1.0f, 0.0f});

    // train the network
    float learning_rate = 0.1f;
    float epochs = 30;

    nn.train(inputs, targets, learning_rate, epochs);

    std::cout << "Neural Network training test passed" <<std::endl;
}

int main(){
    test_nn_forward();
    return 0;
}