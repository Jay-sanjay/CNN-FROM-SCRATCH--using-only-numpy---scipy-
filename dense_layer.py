from layer import Layer
import numpy as np

class Dense(Layer):

    """
    input_size: The number of nodes in the input layer.
    output_size: The number of nodes in the output layer.
    """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) # [output_size x input_size]
        self.bais = np.random.randn(output_size, 1)             # [output_size x 1]

    """
    input: Input to the dense layer
    """
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bais
    

    """
    1. We need to calculate dE/dX as well because it will become the dE/dY for our previous layer.
    2. We need to calculate dE/dW and dE/dB to update weights and biases
    """
    def backward(self, output_gradient, learning_rate):
        weight_gradients = np.dot(output_gradient, self.input.T) # dE/dW
        bias_gradients = output_gradient                         # dE/dB
        self.weights -= learning_rate*weight_gradients           # W = W - learning_rate*dE/dW
        self.bais -= learning_rate*bias_gradients                # B = B - learning_rate*dE/dB
        return np.dot(self.weights.T, output_gradient)           # Y = W.X + B