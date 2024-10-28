import os
import torchvision.transforms as transform
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
import random
import matplotlib.pyplot as plt
import torch.nn as nn


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data(Dataset):
    def __init__(self, folder_path, transformers=None):
        self.transformers = transformers
        self.images = []

        to_tensor = transform.ToTensor()
        for subdir in os.listdir(folder_path):

            image_folder = os.path.join(folder_path, subdir, "images")
            imgs = os.listdir(image_folder)

            dates = {}
            img_tensor = {}
            for i in range(len(imgs)):
                year, month = imgs[i].split("_")[2: 4]
                dates[i] = 12 * int(year) + int(month)
                image = Image.open(os.path.join(image_folder, imgs[i]))
                image = to_tensor(image)
                img_tensor[i] = image
            for i in range(len(imgs) - 1):
                for j in range(i + 1, len(imgs)):
                    self.images.append((img_tensor[i], img_tensor[j], abs(dates[i] - dates[j])))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if self.transformers is None:
            return self.images[idx]

        img1, img2, time_stamp = self.images[idx]
        seed = torch.randint(0, 2 ** 32, (1,)).item()
        random.seed(seed)
        torch.manual_seed(seed)
        img1_transformed = self.transformers(img1)

        random.seed(seed)
        torch.manual_seed(seed)
        img2_transformed = self.transformers(img2)

        return img1_transformed, img2_transformed, time_stamp


transformer = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transform.RandomVerticalFlip(),
    transform.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3))
])
dataset = Data("../Dataset", transformers=transformer)
for data in dataset:
    _, _, t = data

train_size = int(0.7 * len(dataset))
validation_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class ImageModel(nn.Module):
    def __init__(self, time_skip_embedding_dim=16):
        super(ImageModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # [256, 8, 8]
        )

        self.time_embedding = nn.Embedding(30, time_skip_embedding_dim)

        self.fc = nn.Linear(256 * 8 * 8 + time_skip_embedding_dim, 256 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, img, time_skip):
        img_features = self.encoder(img)

        batch_size = img.shape[0]
        img_features_flat = img_features.view(batch_size, -1)
        time_embedding = self.time_embedding(time_skip)

        combined_features = torch.cat((img_features_flat, time_embedding), dim=1)
        combined_features = self.fc(combined_features)

        combined_features = combined_features.view(batch_size, 256, 8, 8)

        output = self.decoder(combined_features)

        return output


def train(model, loader, criterion, optimizer, device=get_default_device()):
    model.train()
    total_loss = 0
    for start_img, end_img, time_skip in loader:
        start_img, end_img, time_skip = start_img.to(device), end_img.to(device), time_skip.to(device)

        optimizer.zero_grad()
        output = model(start_img, time_skip)

        loss = criterion(output, end_img)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device=get_default_device()):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for start_img, end_img, time_skip in loader:
            start_img, end_img, time_skip = start_img.to(device), end_img.to(device), time_skip.to(device)

            output = model(start_img, time_skip)
            loss = criterion(output, end_img)

            total_loss += loss.item()
    return total_loss / len(loader)


def run(n, model, train_loader, validation_loader, criterion, optimizer, device=get_default_device()):
    train_losses = []
    val_losses = []

    for epoch in range(n):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, validation_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{n}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
run(10, model, train_loader, validation_loader, criterion, optimizer, device)
