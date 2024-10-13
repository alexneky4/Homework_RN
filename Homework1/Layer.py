import torch
import math


class Layer:
    def __init__(self, input_size, output_size, activation, derivation, device):
        self.w = self.xavier_initialization(input_size, output_size, device)
        self.b = torch.randn(1, output_size, device=device)
        self.input_data = None
        self.delta = None
        self.activation = activation
        self.derivation = derivation

    def xavier_initialization(self, input_size, output_size, device):
        limit = math.sqrt(6 / (input_size + output_size))
        return torch.randn(input_size, output_size, device=device) * limit

    def feed_forward(self, input_data):
        self.input_data = input_data
        return self.activation(self.input_data @ self.w + self.b)

    def backpropagation(self, delta):
        self.delta = delta
        return self.derivation(self.input_data) * (delta @ self.w.T)

    def update_weights(self, lr=0.001):
        if self.delta is not None:
            self.w -= lr * (self.input_data.T @ self.delta)
            self.b -= lr * self.delta.sum(dim=0)
