import numpy as np
from scipy import signal
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layer import Layer

class Convoultional(Layer):
    """
    input_shape: (height, depth, width) of input
    kernel_size: it is a square matrix, so the isize of that
    depth: number of kernels
    """
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height-kernel_size+1, input_width-kernel_size+1)  
        
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)      # depth: number of kernels of size - {input_depth, kernel_size, kernel_size}

        self.kernels = np.random.randn(*self.kernel_shape) # initailzed the kernels with random values
        self.biases = np.random.randn(*self.output_shape)  # initalized the baises with random values, bais_shape = output_shape

    """
    input: It is 3D

    Yi = Bi + sum{j=1:n ( Xj * K[i][j] )}
    where, * ---> cross correlation (conv) operation

    Used: "valid" for convolution that means the entire kernel is place our the image.
        : "full" In this convolution we do not place the entire kernel on our image.

    """
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        # convolutional operation
        for i in range(self.depth): # for each kernel block
            for j in range(self.input_depth):   # for the first kernel in the block
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        
        return self.output
    
    """
    dE/dK: Derivative wrt kernel
    dE/dX: Derivative wrt input, this is passed as output gradient to the previous layer while back-propagation
   
    dE/dK(i,j)   ===>   X(i) (valid-convolution) dE/dY(j)
    dE/dB(i)     ===>   dE/dY(i)
    dE/dX(j)     ===>   sum(i=1:n) --> dE/dY(i) (full-convolution) K(i,j)
    """
    def backward(self, output_gradient, learning_rate):
        kernel_gradient = np.zeros(self.kernel_shape)
        bais_gradient = np.zeros(self.output_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                
                # dE/dK
                kernel_gradient[i][j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                
                # dE/dX
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        # dE/dB
        bais_gradient = np.copy(output_gradient)


        # Parameter update
        self.kernels -= learning_rate * kernel_gradient
        self.biases  -= learning_rate * bais_gradient

        return input_gradient # this will be used as output gradient for the previous layer