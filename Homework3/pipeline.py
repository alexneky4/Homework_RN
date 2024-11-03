import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import optim
import torch.nn as nn
from torch.optim import lr_scheduler
import os
import importlib.util
import timm
import models
from torch.utils.tensorboard import SummaryWriter
import detectors


class CachedDataset(Dataset):
    def __init__(self, dataset_type, train=True, transform=None):
        dataset_classes = {"MNIST": datasets.MNIST, "CIFAR10": datasets.CIFAR10, "CIFAR100": datasets.CIFAR100}
        if dataset_type not in dataset_classes:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        normalization_values = {
            "MNIST": ([0.1307], [0.3081]),
            "CIFAR10": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
            "CIFAR100": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        }

        transform_list = [transforms.ToTensor()]
        if transform:
            transform_list.append(transform)
        if dataset_type in normalization_values:
            mean, std = normalization_values[dataset_type]
            transform_list.append(transforms.Normalize(mean=mean, std=std))

        self.transform = transforms.Compose(transform_list)
        self.dataset = dataset_classes[dataset_type](root="/dataset", download=True, train=train,
                                                     transform=self.transform)
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]

        return self.cache[idx]


class EarlyStopping:
    def __init__(self, criterion="loss", patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.criterion = criterion
        self.best_score = None

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False
        score_comparison = (current_score < self.best_score) if self.criterion == 'loss' else (
                current_score > self.best_score)
        if self.best_score is None or score_comparison:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs.")
                return True
        return False


def _get_transform(config_transform):
    available_transform = {
        "resize": transforms.Resize,
        "random_crop": transforms.RandomCrop,
        "erase": transforms.RandomErasing,
        "random_vertical_flip": transforms.RandomVerticalFlip,
        "color_jitter": transforms.ColorJitter,
        "random_affine": transforms.RandomAffine,
        "random_horizontal_flip": transforms.RandomHorizontalFlip,
        "random_rotation": transforms.RandomRotation,
        "gaussian_blur": transforms.GaussianBlur,
        "random_erase": transforms.RandomErasing
    }
    transform_list = []
    for transform_name, params in config_transform.items():
        if transform_name in available_transform:
            if params:
                try:
                    transform_list.append(available_transform[transform_name](**params))
                except TypeError as e:
                    print(f"Error initializing {transform_name} with parameters {params}: {e}")
            else:
                transform_list.append(available_transform[transform_name]())
            available_transform.pop(transform_name)
        else:
            print(f"Warning: {transform_name} is not an available transformation. Ignored")
    return transforms.Compose(transform_list)


def load_model_from_path(model_path, model_class_name, *args, **kwargs):
    module_name = os.path.splitext(os.path.basename(model_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    model_class = getattr(model_module, model_class_name)
    model_instance = model_class(*args, **kwargs)
    return model_instance


def get_optimizer(model_params, config_optimizer):
    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "RMSprop": optim.RMSprop
    }

    optim_type = config_optimizer.get("type", "SGD")
    optim_params = config_optimizer.get("params", {})

    if optim_type not in optimizers:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")
    try:
        optimizer = optimizers[optim_type](model_params, **optim_params)
    except TypeError as e:
        raise ValueError(
            f"Error initializing optimizer '{optim_type}' with parameters {optim_params}. "
            f"Check if all parameters are valid for this optimizer. Details: {e}"
        )

    return optimizer


def get_scheduler(optimizer, config_scheduler):
    schedulers = {
        "StepLR": lr_scheduler.StepLR,
        "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
        "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR
    }
    scheduler_type = config_scheduler.get("type")
    if scheduler_type is None:
        return None

    if scheduler_type not in schedulers:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    scheduler_params = config_scheduler.get("params", {})

    try:
        scheduler = schedulers[scheduler_type](optimizer, **scheduler_params)
    except TypeError as e:
        raise ValueError(
            f"Error initializing scheduler '{scheduler_type}' with parameters {scheduler_params}. "
            f"Check if all parameters are valid for this scheduler. Details: {e}"
        )

    return scheduler


def get_lose_function(config_loss):
    loss_functions = {
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "MSELoss": nn.MSELoss,
        "L1Loss": nn.L1Loss,
        "NLLLoss": nn.NLLLoss,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss
    }

    loss_type = config_loss

    if loss_type not in loss_functions:
        raise ValueError(f"Unsupported loss function type: {loss_type}")

    return loss_functions[loss_type]()


def load_model(config_model):
    # if config_model["model_defined"]:
    model_mapping = {
        "resnet18_cifar10": lambda: timm.create_model('resnet18', pretrained=False, num_classes=10),
        "resnet18_cifar100": lambda: timm.create_model("resnet18_cifar100", pretrained=False),
        "PreActResNet18-CIFAR10": lambda: models.PreActResNet18_C(10),
        "PreActResNet18-CIFAR100": lambda: models.PreActResNet18_C(100),
        "MLP": lambda: models.MLP(),
        "LeNet": lambda: models.LeNet()
    }
    return model_mapping[config_model["model_name"]]()


class TrainingPipeline:
    def __init__(self, configuration_path):
        with open(configuration_path) as f:
            self.config = json.load(f)

        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.transform = _get_transform(self.config['transform'])
        self.train_dataset = CachedDataset(self.config['dataset'], train=True, transform=self.transform)
        self.test_dataset = CachedDataset(self.config['dataset'], train=False)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config["batch_size"]["train"] or 64,
                                           shuffle=self.config["shuffle"]["train"] or True)

        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.config["batch_size"]["test"] or 64,
                                          shuffle=self.config["shuffle"]["test"] or True)

        self.model = load_model(self.config["model"])

        self.optimizer = get_optimizer(self.model.parameters(), self.config["optimizer"])
        self.scheduler = get_scheduler(self.optimizer, self.config["scheduler"])
        self.epochs = self.config.get("epochs", 150)

        log_dir = self.config.get("logging", {}).get("log_dir", None)
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

        early_stopping_mechanism = self.config.get("early_stopping", {})
        patience = early_stopping_mechanism.get("patience", 10)
        criterion = early_stopping_mechanism.get("criterion", "loss")
        if criterion != "loss" and criterion != "accuracy":
            print(f"Warning: Criterion {criterion} invalid. Default loss criterion will be used")
        verbose = early_stopping_mechanism.get("verbose", True)
        self.early_stopping = EarlyStopping(patience=patience, criterion=criterion, verbose=verbose)

        self.loss_fn = get_lose_function(self.config["loss"])

    def train(self):
        self.model.to(self.device)

        # history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
        best_train_acc = 0.0
        best_val_acc = 0.0
        scaler = torch.amp.GradScaler("cuda") if self.device.type == 'cuda' else torch.amp.GradScaler()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                with torch.autocast(self.device.type, enabled=self.config.get("use_amp", False)):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                total_loss += loss.item() * images.size(0)
                predicted = outputs.argmax(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            avg_loss = total_loss / total
            acc = 100.0 * correct / total

            if acc > best_train_acc:
                best_train_acc = acc

            # history["train_accuracy"].append(accuracy)

            if self.writer:
                self.writer.add_scalar('Training Loss', avg_loss, epoch)
                self.writer.add_scalar('Training Accuracy', acc, epoch)

            val_loss, val_acc = self.validate(epoch)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # history["val_accuracy"].append(val_accuracy)

            if self.early_stopping(val_loss if self.early_stopping.criterion == 'loss' else val_acc):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        return best_train_acc, best_val_acc

    def validate(self, epoch):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        tta_enabled = self.config.get("use_tta", False)
        tta_repeats = self.config.get("tta_repeats", 5) if tta_enabled else 1
        with torch.no_grad():
            for images, labels in self.test_dataloader:

                images, labels = images.to(self.device), labels.to(self.device)

                if tta_enabled:
                    tta_images = torch.stack([self.transform(image.cpu()) for image in images]).to(self.device)
                    with torch.autocast(self.device.type, enabled=self.config.get("use_amp", False)):
                        averaged_outputs = self.model(tta_images)

                    for _ in range(1, tta_repeats):
                        tta_images = torch.stack([self.transform(image.cpu()) for image in images]).to(self.device)
                        with torch.autocast(self.device.type, enabled=self.config.get("use_amp", False)):
                            outputs = self.model(tta_images)
                            averaged_outputs += outputs
                    averaged_outputs /= tta_repeats
                    loss = self.loss_fn(averaged_outputs, labels)
                else:
                    with torch.autocast(self.device.type, enabled=self.config.get("use_amp", False)):
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, labels)
                    averaged_outputs = outputs
                total_loss += loss.item() * images.size(0)
                predicted = averaged_outputs.argmax(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        acc = 100.0 * correct / total

        if self.writer:
            self.writer.add_scalar("Validation Loss", avg_loss, epoch)
            self.writer.add_scalar("Validation Accuracy", acc, epoch)

        return avg_loss, acc
