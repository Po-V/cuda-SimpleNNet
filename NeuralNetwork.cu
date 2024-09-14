#include "NeuralNetwork.cuh"
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int> &layer_size){
    //size_t is used with size() function for vectors and index operation
    for(size_t i =1; i<layer_size.size(); ++i){
        // create layer object at the end of container
        int activation_type = (i == layer_size.size() - 1) ? 1: 0; // isgmod for last layer, relu for others
        layers.emplace_back(layer_size[i-1], layer_size[i], activation_type);
    }
}

NeuralNetwork::~NeuralNetwork(){
    // cleanup is handled by Layer destructor
}

std::vector<float> NeuralNetwork::forward(const std::vector<float> &input){
    std::vector<float> current_input = input;
    // &layer creates reference to each element in layers rather than making a copy
    for(auto &layer : layers){
        current_input = layer.forward(current_input);
    }
    return current_input;
}

void NeuralNetwork::train(const std::vector<std::vector<float>> & inputs,
                          const std::vector<std::vector<float>> &targets,
                          float learning_rate, int epochs){
    
    for(int epoch = 0; epoch < epochs; ++epoch){

        float total_loss = 0.0f;
        for(size_t i = 0; i < inputs.size(); ++i){
            //forward pass
            std::vector<float> output = forward(inputs[i]);

            // compute loss
            std::vector<float> error(output.size());
            for(size_t j =0; j< output.size(); ++j){
                // targets is vector of vectors and represents multiple sets of target values
                // output is single vector and represents the network's output for 1 input
                error[j] = targets[i][j] - output[j];
                // calculate sum of squared errors
                total_loss += error[j] * error[j];
            }

            // backward pass
            std::vector<float> next_layer_error(output.size(), 0.0f);
            for(int j = layers.size() - 1; j>=0; --j){
                next_layer_error = layers[j].backward(error, learning_rate, (j == layers.size() - 1) ? nullptr : next_layer_error.data());
                error = next_layer_error;
            }
        }
        std::cout<<"Epoch"<<epoch<<", Loss: "<<total_loss / inputs.size()<< std::endl;
        // if(epoch % 1000 == 0){
        //     // total_loss / inputs size to get MSE
        //     std::cout<<"Epoch"<<epoch<<", Loss: "<<total_loss / inputs.size()<< std::endl;
        // }
    }
}