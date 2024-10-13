# Homework 1

The folder contains a Jupyter Notebook with 3 code cells:
1. The first cell retrieves the data from the MNIST dataset.
2. The second cell runs the model, which is structured as a class.
3. The third cell runs the model that does not have a class structure, using only functions.

-[Google Colab link](https://colab.research.google.com/drive/1L2pLkxipPAzf4Jxx47FD4Zl7e9s3CqoB?usp=sharing)
## The Files

### `data.py`
This script retrieves the dataset. It contains helper functions to shape the data and get the default device.
- **Main function**: `load_mnist()`

### `simple_model.py`
This script contains a model structured similarly to the example from the laboratory, modified to work with multiple layers.

- `feed_forward()` stores the outputs of each layer in a list and returns the list. It takes as parameters the input, weights, and biases. All layers use the **sigmoid** function as their activation, except for the last layer, which uses **softmax**.
  
- `backpropagation()` calculates the delta for each layer. The delta for the last layer is `y_predicted - y_labels`, representing the overall error of the model. For previous layers, the delta is calculated as the sigmoid derivative multiplied by the matrix multiplication between the previous delta and the transpose of the previous layer's weights. The gradient is calculated as the matrix multiplication between the transpose of the input and the delta.
  
- `train_batch()` runs a `feed_forward()` followed by `backpropagation()`, then updates and returns the updated weights and biases.
  
- `train_epoch()` runs `train_batch()` for every batch from the input, keeping track of updated weights and biases, and returns them.
  
- `evaluate()` computes the model's accuracy on the test data.

- `train()` runs `train_epoch()` for a given number of iterations, evaluating the model after each epoch.

### `functions.py`
This script contains the following functions:
- `sigmoid()`
- `sigmoid_derivative()`
- `relu()`
- `relu_derivative()`
- `softmax()`

### `Layer.py`
This script contains the `Layer` class. 

- The class is initialized with the number of input neurons, output neurons, the activation function, its derivative, and the device for the tensors.
  
- `feed_forward()` returns the output of the layer.
  
- `backpropagation()` takes the delta from the next layer and returns the delta for the previous layer.
  
- `update_weights()` updates the weights and biases of the layer.

### `Model.py`
This script contains the `Model` class. It creates three layers of sizes `[784, 50], [50, 50], [50, 10]`, following the homework requirements of 784 input neurons, 100 hidden neurons (split across layers), and 10 output neurons. The model also initializes the batch size (default: 200).

- `feed_forward()`, `backpropagation()`, and `update()` are wrapper methods that call the respective methods in the `Layer` class, connecting the layers' outputs and gradients.

- `train_batch()`, `train_epoch()`, `train()`, and `evaluate()` function as described above, adapted for the `Model` class.

## How to Run

1. Run the first code cell in the notebook to retrieve the training and testing data.
2. Depending on which model you want to use, run either the second or third cell for the class-based or function-based model, respectively.

