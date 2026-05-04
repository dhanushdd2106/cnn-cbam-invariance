import sys, os
sys.path.append(os.getcwd())

import torch
import pandas as pd

from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 10)

model.load_state_dict(torch.load("results/models/resnet_cifar.pth"))
model = model.to(device)
model.eval()

# =========================
# Evaluation Function
# =========================
def evaluate(loader):
    preds, labels_all = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            _, p = torch.max(out, 1)

            preds.extend(p.cpu().numpy())
            labels_all.extend(y.numpy())

    acc = sum(p == l for p, l in zip(preds, labels_all)) / len(labels_all)
    prec = precision_score(labels_all, preds, average='macro', zero_division=0)
    rec = recall_score(labels_all, preds, average='macro', zero_division=0)
    f1 = f1_score(labels_all, preds, average='macro', zero_division=0)

    return acc, prec, rec, f1

# =========================
# Base Transform
# =========================
def base():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

# =========================
# Transformations
# =========================
def tx(s):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (s, 0), 1, 0, fill=0)),
        *base().transforms
    ])

def ty(s):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (0, s), 1, 0, fill=0)),
        *base().transforms
    ])

def txy(s):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (s, s), 1, 0, fill=0)),
        *base().transforms
    ])

def rot(a):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.rotate(img, a)),
        *base().transforms
    ])

def flip():
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.hflip(img)),
        *base().transforms
    ])

translation = [2, 5, 8, 10]
rotation = [2, 5, 10, 15, 20, 25]

results = []

# =========================
# Baseline
# =========================
ds = datasets.CIFAR10("data", train=False, download=True, transform=base())
loader = DataLoader(ds, batch_size=64)

acc, p, r, f = evaluate(loader)
results.append({
    "type": "original",
    "level": 0,
    "accuracy": acc,
    "precision": p,
    "recall": r,
    "f1": f
})

# =========================
# Translation
# =========================
for s in translation:
    for name, func in [
        ("translation_x", tx),
        ("translation_y", ty),
        ("translation_xy", txy)
    ]:
        ds = datasets.CIFAR10("data", train=False, download=True, transform=func(s))
        loader = DataLoader(ds, batch_size=64)

        acc, p, r, f = evaluate(loader)

        results.append({
            "type": name,
            "level": s,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f
        })

# =========================
# Rotation
# =========================
for a in rotation:
    ds = datasets.CIFAR10("data", train=False, download=True, transform=rot(a))
    loader = DataLoader(ds, batch_size=64)

    acc, p, r, f = evaluate(loader)

    results.append({
        "type": "rotation",
        "level": a,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f
    })

# =========================
# Flip
# =========================
ds = datasets.CIFAR10("data", train=False, download=True, transform=flip())
loader = DataLoader(ds, batch_size=64)

acc, p, r, f = evaluate(loader)

results.append({
    "type": "flip",
    "level": 1,
    "accuracy": acc,
    "precision": p,
    "recall": r,
    "f1": f
})

# =========================
# Save Results
# =========================
os.makedirs("results/metrics", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv("results/metrics/invariance_cifar_resnet.csv", index=False)

print(df)
print("✅ ResNet CIFAR invariance completed!")