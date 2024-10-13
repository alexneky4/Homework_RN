import torch
from torchvision import datasets
import numpy as np


def collate(x):
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], torch.Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise Exception("Not supported yet")


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')
def to_one_hot(x):
    return torch.eye(x.max() + 1, device=x.device)[x]


def load_mnist(path: str = "./data", train=True):
    mnist_raw = datasets.MNIST(path, download=True, train=train)
    mnist_data = []
    mnist_labels = []
    for image, label in mnist_raw:
        tensor = torch.from_numpy(np.array(image))
        mnist_data.append(tensor)
        mnist_labels.append(label)

    mnist_data = collate(mnist_data).float()  # shape 60000, 28, 28
    mnist_data = mnist_data.flatten(start_dim=1)  # shape 60000, 784
    mnist_data /= mnist_data.max()  # min max normalize
    mnist_labels = collate(mnist_labels)  # shape 60000
    if train:
        mnist_labels = to_one_hot(mnist_labels)
    return mnist_data, mnist_labels
