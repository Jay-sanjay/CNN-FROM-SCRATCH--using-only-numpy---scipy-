class Layer:

    # constructor
    def __init__(self, input, output):
        self.input = None
        self.output = None

    # Forward Propagation
    def forward(self, input):
        pass

    # Backward Propagation
    def backward(self, output_gradient, learning_rate):
        pass