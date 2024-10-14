import torch
from Layer import Layer
import functions
from tqdm import tqdm


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')


class Model:
    def __init__(self, device=get_default_device()):
        self.layers = [
            Layer(784, 50, functions.sigmoid, functions.sigmoid_derivative, device),
            Layer(50, 50, functions.sigmoid, functions.sigmoid_derivative, device),
            Layer(50, 10, functions.softmax, functions.sigmoid_derivative, device),
        ]
        self.device = device
        self.lr = 0.001
        self.batch_size = 200

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x

    def backpropagation(self, y_predicted, y_label):
        delta = y_predicted - y_label
        for layer in reversed(self.layers):
            delta = layer.backpropagation(delta)

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights(self.lr)

    def train_batch(self, x, y):
        output = self.feed_forward(x)
        self.backpropagation(output, y)
        self.update_weights()

    def train_epoch(self, x, y):
        for i in range(0, x.shape[0], self.batch_size):
            x_batch = x[i: i + self.batch_size].to(self.device)
            y_batch = y[i: i + self.batch_size].to(self.device)
            self.train_batch(x_batch, y_batch)

    def train(self, x, y, x_test, y_test, epochs):

        epochs_range = tqdm(range(epochs))
        for e in epochs_range:
            self.train_epoch(x, y)
            acc, loss = self.evaluate(x_test, y_test, 500)
            epochs_range.set_postfix_str(f"Epoch: {e}; Accuracy = {acc:.4f}; Loss = {loss}")

    def evaluate(self, x, y, batch_size):
        total_correct_predictions = 0
        total_loss = 0.0
        total_len = x.shape[0]

        for i in range(0, total_len, batch_size):
            x_batch = x[i: i + batch_size].to(self.device)
            y_batch = y[i: i + batch_size].to(self.device)

            predicted_distribution = self.feed_forward(x_batch)

            _, predicted_max_value_indices = torch.max(predicted_distribution, dim=1)

            correct_predictions = (predicted_max_value_indices == y_batch).sum().item()
            total_correct_predictions += correct_predictions
            total_loss += torch.nn.functional.cross_entropy(predicted_distribution, y_batch, reduction='sum').item()

        return total_correct_predictions / total_len, total_loss / total_len
