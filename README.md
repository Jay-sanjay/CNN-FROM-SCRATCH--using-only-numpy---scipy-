# Convolutional Neural Network from Scratch

This project implements a Convolutional Neural Network (CNN) from scratch using NumPy. The implementation includes all the fundamental building blocks of a CNN and demonstrates its application on the MNIST dataset.

## Project Structure

- `layer.py`: Base layer class definition
- `convolutional.py`: Implementation of convolutional layer
- `activation_layer.py`: Base class for activation layers
- `activations.py`: Implementation of activation functions (Sigmoid, Tanh)
- `dense_layer.py`: Implementation of fully connected layer
- `reshape.py`: Layer for reshaping data between convolution and dense layers
- `losses.py`: Implementation of loss functions (MSE, Binary Cross Entropy)
- `test1.py`: Main script demonstrating CNN usage on MNIST

## Key Components

### 1. Layers
- **Convolutional Layer**: Implements forward and backward propagation for convolution operations
- **Dense Layer**: Fully connected layer implementation
- **Activation Layer**: Supports Sigmoid and Tanh activations
- **Reshape Layer**: Handles data restructuring between layers

### 2. Network Architecture
The implemented CNN architecture includes:
- Input layer (28x28 grayscale images)
- Convolutional layer with 5 kernels (3x3)
- Sigmoid activation
- Reshape layer
- Dense layer (100 nodes)
- Sigmoid activation
- Output layer (10 nodes)
- Sigmoid activation

### 3. Training
- Uses Binary Cross Entropy loss
- Supports mini-batch gradient descent
- Implements backpropagation across all layer types

## Usage

```python
# Example usage
network = [
    Convoultional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5*26*26, 1)),
    Dense(5*26*26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]
```

## Dependencies
- NumPy
- SciPy
- scikit-learn (for MNIST dataset)

## Implementation Details

The network is built using object-oriented principles where each layer implements:
- Forward propagation (`forward()`)
- Backward propagation (`backward()`)
- Parameter updates during training

The convolutional layer implements:
- Cross-correlation for forward pass
- Full convolution for backward pass
- Kernel and bias updates

## Getting Started

### 1. Setting up Python Environment

```bash
# Create a new virtual environment
python -m venv cnn_env

# Activate the virtual environment
# On Windows
cnn_env\Scripts\activate
# On Unix or MacOS
source cnn_env/bin/activate
```

### 2. Installing Dependencies

```bash
# Install required packages
pip install numpy
pip install scipy
pip install scikit-learn
```

### 3. Running the Tests

```bash
# Navigate to the project directory
cd /path/to/CNN/

# Run the test script
python test1.py
```

### 4. Expected Output

The script will:
1. Load and preprocess the MNIST dataset
2. Train the CNN for 50 epochs
3. Display training progress:
   ```
   1/50, error=0.xxxxx
   2/50, error=0.xxxxx
   ...
   ```
4. Show test results:
   ```
   pred: x, true: y
   ...
   Test accuracy: 0.xxxx
   ```

### 5. Troubleshooting

If you encounter any issues:
- Ensure all dependencies are correctly installed
- Check Python version (recommended: Python 3.7+)
- Verify MNIST dataset download (requires internet connection)
- Confirm proper directory structure
