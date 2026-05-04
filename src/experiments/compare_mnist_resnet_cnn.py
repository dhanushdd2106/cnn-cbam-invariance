import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Load Data
# =========================
cnn = pd.read_csv("results/metrics/invariance_full_cnn.csv")
resnet = pd.read_csv("results/metrics/invariance_resnet_mnist.csv")

# =========================
# Create Output Folder
# =========================
os.makedirs("results/comparison", exist_ok=True)

# =========================
# Plot Function
# =========================
def plot_compare(transform):
    cnn_df = cnn[cnn["type"] == transform]
    res_df = resnet[resnet["type"] == transform]

    if cnn_df.empty or res_df.empty:
        return

    plt.figure()
    plt.plot(cnn_df["level"], cnn_df["accuracy"], marker='o', label="CNN")
    plt.plot(res_df["level"], res_df["accuracy"], marker='o', label="ResNet")

    plt.title(f"MNIST Comparison - {transform}")
    plt.xlabel("Level")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(f"results/comparison/resnet_cnn_mnist_{transform}.png")
    plt.close()

# =========================
# Transform Comparisons
# =========================
for t in ["translation", "rotation"]:
    plot_compare(t)

# =========================
# Flip Comparison
# =========================
cnn_flip = cnn[cnn["type"] == "flip"]["accuracy"].values[0]
res_flip = resnet[resnet["type"] == "flip"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "ResNet"], [cnn_flip, res_flip])
plt.title("MNIST Flip Comparison")
plt.ylabel("Accuracy")
plt.savefig("results/comparison/resnet_cnn_mnist_flip.png")
plt.close()

# =========================
# Baseline Comparison
# =========================
cnn_base = cnn[cnn["type"] == "original"]["accuracy"].values[0]
res_base = resnet[resnet["type"] == "original"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "ResNet"], [cnn_base, res_base])
plt.title("MNIST Baseline Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("results/comparison/resnet_cnn_mnist_baseline.png")
plt.close()

# =========================
# Summary Table
# =========================
summary = []

for t in ["original", "translation", "rotation", "flip"]:
    cnn_val = cnn[cnn["type"] == t]["accuracy"].mean()
    res_val = resnet[resnet["type"] == t]["accuracy"].mean()

    summary.append({
        "type": t,
        "cnn_acc": cnn_val,
        "resnet_acc": res_val
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/comparison/resnet_cnn_mnist_summary.csv", index=False)

# =========================
# Print
# =========================
print(summary_df)
print("✅ Final MNIST CNN vs ResNet comparison done!")