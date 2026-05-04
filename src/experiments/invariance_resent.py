import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load ResNet Model
# =========================
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("results/models/resnet_mnist.pth"))
model = model.to(device)
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

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall, f1


# =========================
# Base Transform (IMPORTANT)
# =========================
def base_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])


# =========================
# Transformations
# =========================
def translate_x(shift):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (shift, 0), 1, 0, fill=0)),
        *base_transform().transforms
    ])

def translate_y(shift):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (0, shift), 1, 0, fill=0)),
        *base_transform().transforms
    ])

def translate_xy(shift):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (shift, shift), 1, 0, fill=0)),
        *base_transform().transforms
    ])

def rotate(angle):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.rotate(img, angle)),
        *base_transform().transforms
    ])

def flip():
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.hflip(img)),
        *base_transform().transforms
    ])


translation_levels = [2, 5, 8, 10]
rotation_levels = [2, 5, 10, 15, 20, 25]

results = []

# =========================
# Baseline
# =========================
dataset = datasets.MNIST("data", train=False, download=True, transform=base_transform())
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
# Translation Experiments
# =========================
for shift in translation_levels:
    for name, func in [
        ("translation_x", translate_x),
        ("translation_y", translate_y),
        ("translation_xy", translate_xy)
    ]:
        dataset = datasets.MNIST("data", train=False, download=True, transform=func(shift))
        loader = DataLoader(dataset, batch_size=64)

        acc, prec, rec, f1 = evaluate(loader)

        results.append({
            "type": name,
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