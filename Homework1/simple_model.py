import torch
import math
from tqdm import tqdm


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# device = get_default_device()
# layers_size = [784, 20, 40, 40, 10]
# weights = [torch.randn(layers_size[i], layers_size[i + 1], device=device) for i in range(len(layers_size) - 1)]
# biases = [torch.randn(1, layers_size[i], device=device) for i in range(1, len(layers_size))]

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softmax(x):
    x_exp = torch.exp(x - torch.max(x))
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)


def feed_forward(x, weights, biases):
    outputs = [sigmoid(x @ weights[0] + biases[0])]
    for i in range(1, len(weights) - 1):
        outputs.append(sigmoid(outputs[-1] @ weights[i] + biases[i]))
    outputs.append(softmax(outputs[-1] @ weights[-1] + biases[-1]))
    return outputs


def backpropagation(x, y_labels, outputs, weights, biases):
    y_predicted = outputs[-1]
    delta = [None] * len(weights)

    delta[-1] = y_predicted - y_labels
    for i in range(len(weights) - 2, -1, -1):
        delta[i] = (outputs[i] * (1 - outputs[i])) * (delta[i + 1] @ weights[i + 1].T)

    dW = []
    db = []
    dW.append(x.T @ delta[0])
    db.append(delta[0].mean(dim=0))

    for i in range(1, len(weights)):
        dW.append(outputs[i - 1].T @ delta[i])
        db.append(delta[i].mean(dim=0))

    return dW, db


def xavier_initialization(input_size, output_size, device):
    limit = math.sqrt(6 / (input_size + output_size))
    return torch.randn(input_size, output_size, device=device) * limit


def train_batch(x, y, w, b, lr):
    outputs = feed_forward(x, w, b)
    dW, db = backpropagation(x, y, outputs, w, b)
    for i in range(len(w)):
        w[i] -= lr * dW[i]
        b[i] -= lr * db[i]

    return w, b


def train_epoch(data, labels, w, b, lr, batch_size):
    non_blocking = w[0].device.type == 'cuda'
    for i in range(0, data.shape[0], batch_size):
        x = data[i: i + batch_size].to(w[0].device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w[0].device, non_blocking=non_blocking)
        w, b = train_batch(x, y, w, b, lr)

    return w, b


def train(data, labels, data_test, labels_test, batch_size=100, epochs=1000, device=get_default_device()):
    print(f"Using device {device}")

    layers_size = [784, 50, 50, 10]
    w = [xavier_initialization(layers_size[i], layers_size[i + 1], device=device) for i in range(len(layers_size) - 1)]
    b = [torch.randn(layers_size[i + 1], device=device) for i in range(len(layers_size) - 1)]

    lr = 0.001
    eval_batch_size = 500

    epochs_range = tqdm(range(epochs))

    for e in epochs_range:
        w, b = train_epoch(data, labels, w, b, lr, batch_size)
        accuracy, loss  = evaluate(data_test, labels_test, w, b, eval_batch_size)
        epochs_range.set_postfix_str(f"Epoch: {e}; Accuracy = {accuracy:.4f}; Loss = {loss}")

    return w, b


def evaluate(data, labels, w, b, batch_size):
    total_correct_predictions = 0
    total_len = data.shape[0]
    total_loss = 0.0
    non_blocking = w[0].device.type == 'cuda'

    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(w[0].device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w[0].device, non_blocking=non_blocking)

        predicted_distribution = feed_forward(x, w, b)[-1]

        _, predicted_max_value_indices = torch.max(predicted_distribution, dim=1)
        correct_predictions = (predicted_max_value_indices == y).sum().item()
        total_correct_predictions += correct_predictions
        total_loss += torch.nn.functional.cross_entropy(predicted_distribution, y, reduction='sum').item()

    return total_correct_predictions / total_len, total_loss / total_len

# device = get_default_device()
# train_data, train_labels, test_data, test_labels = load_mnist(device=device)
# w, b = train(train_data, train_labels, 100, 1000, device=device)
#
# acc = evaluate(test_data, test_labels, w, b)
