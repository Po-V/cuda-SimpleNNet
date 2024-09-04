#pragma once

#include <vector>

class Layer{
public:
    Layer(int input_size, int output_size, int activation_type);
    ~Layer();

    std::vector<float> forward(const std::vector<float> &input);
    std::vector<float>backward(const std::vector<float> &output_error, float learning_rate, float* next_layer_error = nullptr);

private:
    int input_size;
    int output_size;
    int activation_type;
    float* weights_d;
    float* bias_d;
    float* input_d;
    float* output_d;
    float* output_error_d;
    float* input_error_d;
};