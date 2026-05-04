import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Transform (IMPORTANT)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),     # ResNet requirement
    transforms.Grayscale(num_output_channels=3),  # convert 1 → 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# =========================
# Model
# =========================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# =========================
# Training
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

epochs = 5
best_acc = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)

            total += y.size(0)
            correct += (pred == y).sum().item()

    acc = correct / total
    print(f"Epoch {epoch+1}, Acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        os.makedirs("results/models", exist_ok=True)
        torch.save(model.state_dict(), "results/models/resnet_mnist.pth")

print("✅ ResNet MNIST training done!")