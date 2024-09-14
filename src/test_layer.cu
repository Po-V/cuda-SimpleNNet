#include "layer.cuh"
#include <cassert>
#include <iostream>

void test_layer_forward(){

    Layer layer(20, 5, 0);

    std::vector<float> input(10, 0.5f);

    std::vector<float> output = layer.forward(input);

    assert(output.size() == 5);

    for(float val : output){
        assert(val >= 0.0f);
    }

    std::cout << "Layer forward test passed" << std::endl;
}


void test_layer_backward(){

    Layer layer(10, 2, 1);

    std::vector <float> input(10, 0.3f);
    std::vector<float> next_layer_error(2, 0.1f);

    layer.forward(input);

    std::vector<float> input_error = layer.backward(next_layer_error, 0.01f);

    assert(input_error.size() == 10);

    std::cout << "Layer backward test passed" <<std::endl;
}

int main(){
    test_layer_forward();
    test_layer_backward();

    return 0;
}