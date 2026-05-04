import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Data
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10("data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# =========================
# Model (ResNet)
# =========================
model = models.resnet18(pretrained=True)

# Modify final layer
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

epochs = 12

train_losses = []
test_accuracies = []

best_acc = 0

# =========================
# Training Loop
# =========================
for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # =========================
    # Evaluation
    # =========================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    test_accuracies.append(acc)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")

    # =========================
    # Save Best Model
    # =========================
    if acc > best_acc:
        best_acc = acc
        os.makedirs("results/models", exist_ok=True)
        torch.save(model.state_dict(), "results/models/resnet_cifar.pth")

# =========================
# Save Plots
# =========================
os.makedirs("results/plots", exist_ok=True)

plt.figure()
plt.plot(train_losses)
plt.title("ResNet CIFAR Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("results/plots/resnet_cifar_loss.png")
plt.close()

plt.figure()
plt.plot(test_accuracies)
plt.title("ResNet CIFAR Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("results/plots/resnet_cifar_accuracy.png")
plt.close()

print("✅ ResNet CIFAR training complete!")