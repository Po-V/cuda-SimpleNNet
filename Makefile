# # Compiler
# NVCC := nvcc
# # Compiler flags
# NVCC_FLAGS := -std=c++11 -O3 -arch=sm_75

# # source files
# SRC := main.cu NeuralNetwork.cu layer.cu utils.cu
# # object files
# OBJ := $(SRC:.cu=.o)

# # Test source files
# TEST_SRC := test_layer.cu test_neural_network.cu
# # test object files
# TEST_OBJ := $(TEST_SRC:.cu=.o)
# TEST_EXE := $(TEST_SRC:.cu=_test)

# # target executable
# TARGET := NeuralNetwork
# # TEST_TARGET := Tests

# # default target
# all: $(TARGET) $(TEST_EXE)

# # Link the target executable
# $(TARGET): $(OBJ)
# 	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(TARGET)

# $(TEST_EXE): %_test: %.o
# 	$(NVCC) $(NVCC_FLAGS) $< -o $@

# # Link the test executable
# # $(TEST_TARGET): $(TEST_OBJ)
# # 	$(NVCC) $(NVCC_FLAGS) $(TEST_OBJ) -o $(TEST_TARGET)

# # compile source files
# %.o: %.cu
# 	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# %_test.o: %.cu
# 	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# # clean up
# clean:
# 	rm -f $(OBJ) $(TEST_OBJ) $(TEST_EXE) $(TARGET)


# Define the CUDA compiler
NVCC = nvcc

# Define the CUDA compilation flags
NVCC_FLAGS := -std=c++11 -O3 -arch=sm_75

# Define the source files and object files
SRC = layer.cu NeuralNetwork.cu utils.cu
OBJ = layer.o neuralnetwork.o utils.o

# Define the output binaries
TEST_LAYER_EXEC = test_layer
TEST_NEURAL_NETWORK_EXEC = test_neural_network

# Default target to build both test executables
all: $(TEST_LAYER_EXEC) $(TEST_NEURAL_NETWORK_EXEC)

# Rule to build test_layer executable
$(TEST_LAYER_EXEC): test_layer.o layer.o utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ test_layer.o layer.o utils.o

# Rule to build test_neural_network executable
$(TEST_NEURAL_NETWORK_EXEC): test_neural_network.o neuralnetwork.o utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ test_neural_network.o neuralnetwork.o utils.o

# Rule to build object files from .cu files
test_layer.o: test_layer.cu layer.o utils.o
	$(NVCC) $(NVCC_FLAGS) -c test_layer.cu -o test_layer.o

test_neural_network.o: test_neural_network.cu neuralnetwork.o utils.o
	$(NVCC) $(NVCC_FLAGS) -c test_neural_network.cu -o test_neural_network.o

layer.o: layer.cu layer.cuh
	$(NVCC) $(NVCC_FLAGS) -c layer.cu -o layer.o

neuralnetwork.o: NeuralNetwork.cu NeuralNetwork.cuh
	$(NVCC) $(NVCC_FLAGS) -c NeuralNetwork.cu -o neuralnetwork.o

utils.o: utils.cu
	$(NVCC) $(NVCC_FLAGS) -c utils.cu -o utils.o

# Clean rule to remove compiled files
clean:
	rm -f *.o $(TEST_LAYER_EXEC) $(TEST_NEURAL_NETWORK_EXEC)

.PHONY: all clean

