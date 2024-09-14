#include <vector>
#include "NeuralNetwork.cuh"
#include <iostream>
#include <ctime>
#include <random>

void generate_synthetic_data(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, int num_samples){
    // convert type of time_t to usinged int and pass it to gen to set starting point (seed)
    std::mt19937 gen(static_cast<unsigned int>(time(0)));
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(int i =0; i <num_samples; ++i){
        std::vector<float> input(10);

        for(int j =0; j <10; ++j){
            input[j] = dis(gen);
        }

        float sum = 0;
        for(float val : input){
            sum += val;
        }

        inputs.push_back(input);
        // if sum of inputs > 5, it's class 1, otherwise class 0
        if(sum > 5){
            targets.push_back({1.0f, 0.0f});
        } else{
            targets.push_back({0.0f, 1.0f});
        }
    }
}

int main(){

    std::vector<int> layer_size = {10, 50, 2};
    NeuralNetwork nn(layer_size);

    // training data
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    int num_samples = 50;
    generate_synthetic_data(inputs, targets, num_samples);

    float learning_rate = 0.001f;
    int epochs = 100;

    // Train the network
    nn.train(inputs, targets, learning_rate, epochs);

    // Test the network on a few samples
    std::cout << "Testing the network:\n";
    for(int i = 0; i < 5; ++i){
        std::vector<float> output = nn.forward(inputs[i]);
        std::cout << "Input: ";
        for(float val: inputs[i]){
            std::cout <<val <<" ";
        }
        std::cout << "\nTrue label: " << (targets[i][0] > targets[i][1] ? "Class 1" : "Class 0");
        std::cout << "\nOutput: Class 1 probability: " << output[0] << ", Class 2 probability: " << output[1] << std::endl;
        std::cout << "Predicted class: " << (output[0] > output[1] ? "Class 1" : "Class 0") << "\n\n";
    }

    return 0;
}