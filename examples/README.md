# YANNP Examples

This directory contains examples demonstrating the capabilities of the YANNP (Yet Another Neural Network Package) library.

## Available Examples

### 🧠 Neural Network Examples

#### 1. **Basic MLP** (`basic_mlp_v2/`)
- Multi-Layer Perceptron solving the XOR problem
- Demonstrates basic neural network training and prediction
- Uses sigmoid activation with detailed step-by-step output
- Perfect for understanding neural network fundamentals

#### 2. **Enhanced MLP** (`enhanced_mlp/`)
- Advanced Multi-Layer Perceptron with modern features
- Demonstrates:
  - Cross-entropy loss function
  - Adam optimizer
  - He Normal weight initialization
  - L2 regularization
  - Dropout
  - Batch normalization
  - Training with validation

### 🤖 Transformer Examples

#### 3. **ChatGPT** (`chatgpt/`)
- Large-scale transformer model example
- Demonstrates model loading and text generation
- Uses safetensors format for model weights
- Interactive text generation with real pre-trained models

## Running Examples

Each example is a standalone Go program. To run any example:

```bash
cd examples/<example_name>
go run main.go
```

Or build and run:

```bash
cd examples/<example_name>
go build
./<example_name>
```

## Example Categories

### 🟢 Beginner-Friendly
- `basic_mlp_v2/`

### 🟡 Intermediate
- `enhanced_mlp/`

### 🔴 Advanced
- `chatgpt/`

## Key Features Demonstrated

- **Basic Neural Networks**: Forward propagation, backpropagation, training
- **Advanced Features**: Custom loss functions, optimizers, initializers
- **Regularization**: Dropout, batch normalization, L2 regularization
- **Training Strategies**: Validation, different optimizers
- **Model Architectures**: MLPs, Transformers
- **Data Handling**: Various datasets and preprocessing techniques

## Next Steps

After running these examples, you can:

1. **Modify parameters**: Change learning rates, network sizes, epochs
2. **Try different datasets**: Use your own data with the examples
3. **Combine techniques**: Mix different optimizers with regularization
4. **Build custom models**: Use the library components to create new architectures
5. **Experiment with transformers**: Try different prompts and generation parameters

## Dependencies

All examples use the YANNP library and its dependencies:
- `gonum.org/v1/gonum/mat` - Matrix operations
- Various YANNP shared packages (loss, optimizer, regularization, etc.)

## Contributing

To add a new example:

1. Create a new directory under `examples/`
2. Add a `main.go` file with your example
3. Update this README with the new example description
4. Ensure the example is well-documented and follows the existing style

Happy learning! 🎉