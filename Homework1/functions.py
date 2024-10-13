import torch


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    x_exp = torch.exp(x - torch.max(x))
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)


def relu(x):
    return torch.maximum(torch.tensor(0.0, device=x.device), x)


# Derivative of ReLU function
def relu_derivative(x):
    return (x > 0).float()