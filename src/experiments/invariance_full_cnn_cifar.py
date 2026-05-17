import sys, os
sys.path.append(os.getcwd())

import torch
import pandas as pd

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from src.models.cnn_cifar import CNN_CIFAR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
model = CNN_CIFAR().to(device)
model.load_state_dict(torch.load("results/models/cnn_cifar.pth"))
model.eval()

# =========================
# Evaluation
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
# Base transform
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
        transforms.Lambda(lambda img: TF.affine(img, 0, (s,0), 1, 0, fill=0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

def ty(s):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (0,s), 1, 0, fill=0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

def txy(s):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(img, 0, (s,s), 1, 0, fill=0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

def rot(a):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.rotate(img, a)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

def flip():
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.hflip(img)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    
def vflip():
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.vflip(img)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

translation = [2,5,8,10,12,15,18,20,22,25,28,30,32,35,38,40]
rotation = [2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]

results = []

# =========================
# Baseline
# =========================
ds = datasets.CIFAR10("data", train=False, download=True, transform=base())
loader = DataLoader(ds, batch_size=64)

acc, p, r, f = evaluate(loader)
results.append({"type":"original","level":0,"accuracy":acc,"precision":p,"recall":r,"f1":f})

# =========================
# Translation Experiments
# =========================
for s in translation:
    for name, func in [("translation_x",tx),("translation_y",ty),("translation_xy",txy)]:
        ds = datasets.CIFAR10("data", train=False, download=True, transform=func(s))
        loader = DataLoader(ds, batch_size=64)

        acc,p,r,f = evaluate(loader)
        results.append({"type":name,"level":s,"accuracy":acc,"precision":p,"recall":r,"f1":f})

# =========================
# Rotation
# =========================
for a in rotation:
    ds = datasets.CIFAR10("data", train=False, download=True, transform=rot(a))
    loader = DataLoader(ds, batch_size=64)

    acc,p,r,f = evaluate(loader)
    results.append({"type":"rotation","level":a,"accuracy":acc,"precision":p,"recall":r,"f1":f})

# =========================
# Flip
# =========================
ds = datasets.CIFAR10("data", train=False, download=True, transform=flip())
loader = DataLoader(ds, batch_size=64)

acc,p,r,f = evaluate(loader)
results.append({"type":"flip","level":1,"accuracy":acc,"precision":p,"recall":r,"f1":f})

# =========================
# Save Results
# =========================
os.makedirs("results/metrics", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv("results/metrics/invariance_cifar_cnn.csv", index=False)

print(df)
print("✅ CNN CIFAR invariance done")