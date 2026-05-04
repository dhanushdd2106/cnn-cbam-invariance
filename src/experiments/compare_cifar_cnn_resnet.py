import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Load Results
# =========================
cnn = pd.read_csv("results/metrics/invariance_cifar_cnn.csv")
resnet = pd.read_csv("results/metrics/invariance_cifar_resnet.csv")

os.makedirs("results/comparison", exist_ok=True)

# =========================
# Plot Function
# =========================
def plot_compare(transform, metric="accuracy"):
    c = cnn[cnn["type"] == transform]
    r = resnet[resnet["type"] == transform]

    if c.empty or r.empty:
        return

    plt.figure()
    plt.plot(c["level"], c[metric], marker='o', label="CNN")
    plt.plot(r["level"], r[metric], marker='o', label="ResNet")

    plt.title(f"CIFAR Comparison - {transform} ({metric})")
    plt.xlabel("Level")
    plt.ylabel(metric)
    plt.legend()

    plt.savefig(f"results/comparison/resnet_cnn_cifar_{transform}_{metric}.png")
    plt.close()

# =========================
# Transform Comparisons
# =========================
for t in ["translation_x", "translation_y", "translation_xy", "rotation"]:
    for m in ["accuracy", "precision", "recall", "f1"]:
        plot_compare(t, m)

# =========================
# Flip Comparison
# =========================
cnn_flip = cnn[cnn["type"] == "flip"]["accuracy"].values[0]
res_flip = resnet[resnet["type"] == "flip"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "ResNet"], [cnn_flip, res_flip])
plt.title("Flip Comparison (CIFAR)")
plt.ylabel("Accuracy")
plt.savefig("results/comparison/resnet_cnn_cifar_flip.png")
plt.close()

# =========================
# Baseline Comparison
# =========================
cnn_base = cnn[cnn["type"] == "original"]["accuracy"].values[0]
res_base = resnet[resnet["type"] == "original"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "ResNet"], [cnn_base, res_base])
plt.title("Baseline Accuracy (CIFAR)")
plt.ylabel("Accuracy")
plt.savefig("results/comparison/resnet_cnn_cifar_baseline.png")
plt.close()

# =========================
# Summary Table
# =========================
summary = []

for t in ["original", "translation_x", "translation_y", "translation_xy", "rotation", "flip"]:
    cnn_val = cnn[cnn["type"] == t]["accuracy"].mean()
    res_val = resnet[resnet["type"] == t]["accuracy"].mean()

    summary.append({
        "type": t,
        "cnn_acc": cnn_val,
        "resnet_acc": res_val
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/comparison/resnet_cnn_cifar_summary.csv", index=False)

print(summary_df)
print("✅ CIFAR CNN vs ResNet comparison completed!")