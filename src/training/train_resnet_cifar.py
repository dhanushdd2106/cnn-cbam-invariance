import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torchvision.datasets as datasets

# Force alternate mirror
datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
datasets.CIFAR10.filename = "cifar-10-python.tar.gz"
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Data (same as CNN)
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

train_dataset = datasets.CIFAR10("data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# =========================
# Model
# =========================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

epochs = 10
train_losses = []
test_accs = []
best_acc = 0

# =========================
# Training Loop
# =========================
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

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

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
    test_accs.append(acc)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        os.makedirs("results/models", exist_ok=True)
        torch.save(model.state_dict(), "results/models/resnet_cifar.pth")

# =========================
# Plots
# =========================
os.makedirs("results/plots", exist_ok=True)

plt.plot(train_losses)
plt.title("ResNet CIFAR Loss")
plt.savefig("results/plots/resnet_cifar_loss.png")
plt.close()

plt.plot(test_accs)
plt.title("ResNet CIFAR Accuracy")
plt.savefig("results/plots/resnet_cifar_accuracy.png")
plt.close()

print("✅ ResNet CIFAR training complete!")