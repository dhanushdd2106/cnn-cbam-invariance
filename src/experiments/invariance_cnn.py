import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from src.models.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("results/models/cnn.pth"))
model.eval()

# Evaluation function
def evaluate(model, loader, device):
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


# Transformations
def get_translation_transform(shift):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, angle=0, translate=(shift, shift), scale=1.0, shear=0)),
        transforms.ToTensor()
    ])

def get_rotation_transform(angle):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.rotate(img, angle)),
        transforms.ToTensor()
    ])

def get_flip_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.hflip(img)),
        transforms.ToTensor()
    ])


translation_levels = [2, 5, 8, 10]
rotation_levels = [2, 5, 10, 15, 20, 25]

results = []

# Baseline
base_transform = transforms.ToTensor()
dataset = datasets.MNIST("data", train=False, download=True, transform=base_transform)
loader = DataLoader(dataset, batch_size=64)

acc, prec, rec, f1 = evaluate(model, loader, device)

results.append({
    "type": "original",
    "level": 0,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1
})

# Translation
for shift in translation_levels:
    transform = get_translation_transform(shift)
    dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64)

    acc, prec, rec, f1 = evaluate(model, loader, device)

    results.append({
        "type": "translation",
        "level": shift,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

# Rotation
for angle in rotation_levels:
    transform = get_rotation_transform(angle)
    dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64)

    acc, prec, rec, f1 = evaluate(model, loader, device)

    results.append({
        "type": "rotation",
        "level": angle,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

# Flip
flip_transform = get_flip_transform()
dataset = datasets.MNIST("data", train=False, download=True, transform=flip_transform)
loader = DataLoader(dataset, batch_size=64)

acc, prec, rec, f1 = evaluate(model, loader, device)

results.append({
    "type": "flip",
    "level": 1,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1
})

# Save results
os.makedirs("results/metrics", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/metrics/invariance_cnn.csv", index=False)

print(df)

# Plot translation
df_t = df[df["type"] == "translation"]
plt.figure()
plt.plot(df_t["level"], df_t["accuracy"], marker='o')
plt.title("CNN Translation Invariance")
plt.xlabel("Shift (pixels)")
plt.ylabel("Accuracy")
plt.savefig("results/plots/cnn_translation.png")

# Plot rotation
df_r = df[df["type"] == "rotation"]
plt.figure()
plt.plot(df_r["level"], df_r["accuracy"], marker='o')
plt.title("CNN Rotation Invariance")
plt.xlabel("Angle (degrees)")
plt.ylabel("Accuracy")
plt.savefig("results/plots/cnn_rotation.png")

# Plot flip
df_f = df[df["type"] == "flip"]
plt.figure()
plt.bar(["Flip"], df_f["accuracy"])
plt.title("CNN Flip Robustness")
plt.ylabel("Accuracy")
plt.savefig("results/plots/cnn_flip.png")
