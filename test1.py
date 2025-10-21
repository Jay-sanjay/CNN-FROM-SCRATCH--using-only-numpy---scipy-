import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convolutional import Convoultional
from reshape import Reshape

from NN.activations import Sigmoid
from NN.dense_layer import Dense
from NN.losses import binary_cross_entropy, binary_cross_entropy_prime
from sklearn.datasets import fetch_openml


def preprocess_data(x, y, limit):
    # First convert y to integers to ensure comparison works correctly
    y_int = y.astype(int)
    
    # Find indices for all digits from 0 to 9
    all_indices = []
    for digit in range(10):  # 0 to 9
        digit_indices = np.where(y_int == digit)[0][:limit]
        all_indices.append(digit_indices)
    
    # Concatenate all indices and shuffle
    all_indices = np.hstack(all_indices)
    all_indices = np.random.permutation(all_indices)
    
    # Select data using these indices
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255 # normalize
    
    # Create one-hot encodings for all 10 classes
    y_int = y.astype(int)
    y_categorical = np.eye(10)[y_int]
    y_categorical = y_categorical.reshape(len(y_categorical), 10, 1)
    
    return x, y_categorical


# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Split into training and test sets
x_train, x_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)


# Model
network = [
    Convoultional((1, 28, 28), 3, 5), # 5 kernels of (5 x 5)
    Sigmoid(), # activating results with sigmoid
    Reshape((5, 26, 26), (5*26*26, 1)), # this is like stacking all in one column
    Dense(5*26*26, 100), # 100 nodes in this dense layer
    Sigmoid(), # activating results with sigmoid
    Dense(100, 10),
    Sigmoid()
]

# training
epochs = 50
learning_rate = 0.01

for e in range(epochs):
    error = 0

    # Process full data to get the error accross all data-points
    for x, y in zip(x_train, y_train):

        # Forward Prop
        output = x
        for layer in network:
            output = layer.forward(output)

        # error
        error += binary_cross_entropy(y, output)

        # Backward
        grad = binary_cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error = error / len(x_train)
    print('%d/%d,  error=%f' % (e+1, epochs, error))


# test
test_accuracy = 0
for x, y in zip(x_test, y_test):
    result = x
    for layer in network:
        result = layer.forward(result)  # Use result instead of output
    print(f"pred: {np.argmax(result)}, true: {np.argmax(y)}")
    if(np.argmax(result) == np.argmax(y)):
        test_accuracy += 1
print(f"Test accuracy: {test_accuracy/len(x_test)}")