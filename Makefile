# Compiler
NVCC = nvcc
# Compiler flags
CXXFLAGS = -std=c++11 -O3 -arch=sm_75

# Directories
SRC_DIR = src
TEST_DIR = tests
OBJ_DIR = obj
BIN_DIR = bin

# Executables
EXEC = $(BIN_DIR)/main
TEST_EXEC = $(BIN_DIR)/tests

#Source files
SRC_FILES = $(SRC_DIR)/main.cu $(SRC_DIR)/NeuralNetwork.cu $(SRC_DIR)/layer.cu
TEST_FILES = $(TEST_DIR)/test_neural_network.cu $(TEST_DIR)/test_layer.cu

# Object files
OBJ_FILES = $(addprefix $(OBJ_DIR)/, $(notdir $(SRC_FILES:.cu=.o)))
TEST_OBJ_FILES = $(addprefix $(OBJ_DIR)/, $(notdir $(TEST_FILES:.cu=.o)))

# Targets
.PHONY: all clean run test

all: $(EXEC)

# Create binary for main executables
$(EXEC): $(OBJ_FILES) | $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^

# Create object files from source
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c -o $@ $<

# Run the main program
run: $(EXEC)
	./$(EXEC)

# Compile and run tests
test: $(TEST_EXEC)
	./$(TEST_EXEC)

# Create binary for the test executable
$(TEST_EXEC): $(TEST_OBJ_FILES) $(OBJ_FILES) | $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^

# Create object files from test source
$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c -o $@ $<

# Create required directories
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

