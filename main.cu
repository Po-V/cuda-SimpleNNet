#include <vector>
#include "NeuralNetwork.cuh"
#include <iostream>

void generate_synthetic_data(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, int num_samples){
    // convert type of time_t to usinged int and pass it to srand to set starting point (seed)
    srand(static_cast<unsigned int>(time(0)));

    for(int i =0; i <num_samples; ++i){
        std::vector<float> input(10);

        for(int j =0; j <10; ++j){
            input[j] = static_cast<float>(rand()) / RAND_MAX;
        }

        // rand() returns int value, used to generate binary outcome
        if(rand() % 2 == 0){
            // add elements to end of vector
            inputs.push_back(input);
            targets.push_back({1.0f, 0.0f});
        }else{
            inputs.push_back(input);
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

    float learning_rate = 0.1f;
    int epochs = 20;

    nn.train(inputs, targets, learning_rate, epochs);

    for(const auto& input : inputs){
        std::vector<float> output = nn.forward(input);
        std::cout <<"Input: ";
        for(float val: input){
            std::cout <<val <<" ";
        }
        std::cout << " Output: Cat probability: " <<output[0] << ", Dog probability: " << output[1] << std::endl;
    }

    return 0;
}