import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models.cnn_cbam import CNN_CBAM

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Data
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# =========================
# Model
# =========================
model = CNN_CBAM().to(device)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# Training
# =========================
epochs = 5

train_losses = []
test_accuracies = []

best_acc = 0

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

    accuracy = correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    # Save best model
    if accuracy > best_acc:
        best_acc = accuracy
        os.makedirs("results/models", exist_ok=True)
        torch.save(model.state_dict(), "results/models/cnn_cbam.pth")

# =========================
# Plot Training
# =========================
os.makedirs("results/plots", exist_ok=True)

plt.figure()
plt.plot(train_losses, label="Loss")
plt.title("Training Loss (CBAM)")
plt.savefig("results/plots/cbam_loss.png")
plt.close()

plt.figure()
plt.plot(test_accuracies, label="Accuracy")
plt.title("Test Accuracy (CBAM)")
plt.savefig("results/plots/cbam_accuracy.png")
plt.close()

print("✅ Training complete and model saved!")