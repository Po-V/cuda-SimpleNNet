# Compiler
NVCC := nvcc
# Compiler flags
NVCC_FLAGS := -std=c++11 -O3 -arch=sm_75

# source files
SRC := main.cu NeuralNetwork.cu layer.cu utils.cu
# object files
OBJ := $(SRC:.cu=.o)
# target executable
TARGET := NeuralNetwork

# default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(TARGET)

# compile source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# clean up
clean:
	rm -f $(OBJ) $(TARGET)

