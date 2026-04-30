import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from src.models.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
model = CNN().to(device)
model.load_state_dict(torch.load("results/models/cnn.pth"))
model.eval()

# =========================
# Evaluation Function
# =========================
def evaluate(loader):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall, f1


# =========================
# Transformations
# =========================

# Horizontal shift
def translate_x(shift):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (shift, 0), 1, 0, fill=0)),
        transforms.ToTensor()
    ])

# Vertical shift
def translate_y(shift):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (0, shift), 1, 0, fill=0)),
        transforms.ToTensor()
    ])

# Diagonal shift
def translate_xy(shift):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (shift, shift), 1, 0, fill=0)),
        transforms.ToTensor()
    ])

# Rotation
def rotate(angle):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.rotate(img, angle)),
        transforms.ToTensor()
    ])

# Flip
def flip():
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.hflip(img)),
        transforms.ToTensor()
    ])


translation_levels = [2, 5, 8, 10]
rotation_levels = [2, 5, 10, 15, 20, 25]

results = []

# =========================
# Baseline
# =========================
dataset = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=64)

acc, prec, rec, f1 = evaluate(loader)

results.append({
    "type": "original",
    "level": 0,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1
})


# =========================
# Translation X
# =========================
for shift in translation_levels:
    dataset = datasets.MNIST("data", train=False, download=True, transform=translate_x(shift))
    loader = DataLoader(dataset, batch_size=64)

    acc, prec, rec, f1 = evaluate(loader)

    results.append({
        "type": "translation_x",
        "level": shift,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })


# =========================
# Translation Y
# =========================
for shift in translation_levels:
    dataset = datasets.MNIST("data", train=False, download=True, transform=translate_y(shift))
    loader = DataLoader(dataset, batch_size=64)

    acc, prec, rec, f1 = evaluate(loader)

    results.append({
        "type": "translation_y",
        "level": shift,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })


# =========================
# Translation XY
# =========================
for shift in translation_levels:
    dataset = datasets.MNIST("data", train=False, download=True, transform=translate_xy(shift))
    loader = DataLoader(dataset, batch_size=64)

    acc, prec, rec, f1 = evaluate(loader)

    results.append({
        "type": "translation_xy",
        "level": shift,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })


# =========================
# Rotation
# =========================
for angle in rotation_levels:
    dataset = datasets.MNIST("data", train=False, download=True, transform=rotate(angle))
    loader = DataLoader(dataset, batch_size=64)

    acc, prec, rec, f1 = evaluate(loader)

    results.append({
        "type": "rotation",
        "level": angle,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })


# =========================
# Flip
# =========================
dataset = datasets.MNIST("data", train=False, download=True, transform=flip())
loader = DataLoader(dataset, batch_size=64)

acc, prec, rec, f1 = evaluate(loader)

results.append({
    "type": "flip",
    "level": 1,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1
})


# =========================
# Save Results
# =========================
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv("results/metrics/invariance_full_cnn.csv", index=False)

print(df)


# =========================
# Plot Translation X
# =========================
plt.figure()
df[df["type"] == "translation_x"].plot(x="level", y="accuracy", marker='o')
plt.title("Translation X (CNN)")
plt.savefig("results/plots/translation_x.png")
plt.close()

# =========================
# Plot Translation Y
# =========================
plt.figure()
df[df["type"] == "translation_y"].plot(x="level", y="accuracy", marker='o')
plt.title("Translation Y (CNN)")
plt.savefig("results/plots/translation_y.png")
plt.close()

# =========================
# Plot Translation XY
# =========================
plt.figure()
df[df["type"] == "translation_xy"].plot(x="level", y="accuracy", marker='o')
plt.title("Translation XY (CNN)")
plt.savefig("results/plots/translation_xy.png")
plt.close()

# =========================
# Plot Rotation
# =========================
plt.figure()
df[df["type"] == "rotation"].plot(x="level", y="accuracy", marker='o')
plt.title("Rotation (CNN)")
plt.savefig("results/plots/rotation.png")
plt.close()

# =========================
# Plot Flip
# =========================
flip_acc = df[df["type"] == "flip"]["accuracy"].values[0]

plt.figure()
plt.bar(["Flip"], [flip_acc])
plt.title("Flip (CNN)")
plt.savefig("results/plots/flip.png")
plt.close()

print("✅ All experiments completed and saved!")