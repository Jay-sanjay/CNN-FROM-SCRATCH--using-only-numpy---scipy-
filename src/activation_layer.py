from src.layer import Layer
import numpy as np

# This class just applies the act. function ---> Y = f(X)
# And also the derivative.
class Activation(Layer):
    """
    activation: The actual Activation Function.
    activation_prime: The derivative of Activation Function.

    Y = f(X)
    """
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    """
    dE/dX = [dE/dY (element_wise_multiply) f'(X)]
    """
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))