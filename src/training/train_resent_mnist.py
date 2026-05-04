import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Config
# =========================
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.0003
FREEZE_BACKBONE = True   # 🔥 keep True for speed

# =========================
# Data (IMPORTANT FIX)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),             # ResNet input
    transforms.Grayscale(num_output_channels=3),  # 1 → 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

# =========================
# Model
# =========================
model = models.resnet18(pretrained=True)

# Modify final layer
model.fc = nn.Linear(model.fc.in_features, 10)

# Freeze backbone (FAST)
if FREEZE_BACKBONE:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

model = model.to(device)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

# =========================
# Tracking
# =========================
train_losses = []
test_accuracies = []
best_acc = 0

# =========================
# Training Loop
# =========================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

    # Save best model
    if acc > best_acc:
        best_acc = acc
        os.makedirs("results/models", exist_ok=True)
        torch.save(model.state_dict(), "results/models/resnet_mnist.pth")

# =========================
# Save Training Logs
# =========================
os.makedirs("results/metrics", exist_ok=True)

df = pd.DataFrame({
    "epoch": list(range(1, EPOCHS+1)),
    "loss": train_losses,
    "accuracy": test_accuracies
})

df.to_csv("results/metrics/resnet_mnist_training.csv", index=False)

# =========================
# Save Plots
# =========================
os.makedirs("results/plots", exist_ok=True)

plt.figure()
plt.plot(train_losses)
plt.title("ResNet MNIST Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("results/plots/resnet_mnist_loss.png")
plt.close()

plt.figure()
plt.plot(test_accuracies)
plt.title("ResNet MNIST Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("results/plots/resnet_mnist_accuracy.png")
plt.close()

print("✅ ResNet MNIST training complete!")