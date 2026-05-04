import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Load Results
# =========================
cnn = pd.read_csv("results/metrics/invariance_full_cnn.csv")
resnet = pd.read_csv("results/metrics/invariance_resnet_mnist.csv")

os.makedirs("results/comparison", exist_ok=True)

# =========================
# Helper Function
# =========================
def plot_compare(transform_type, metric="accuracy"):
    cnn_df = cnn[cnn["type"] == transform_type]
    res_df = resnet[resnet["type"] == transform_type]

    if cnn_df.empty or res_df.empty:
        return

    plt.figure()
    plt.plot(cnn_df["level"], cnn_df[metric], marker='o', label="CNN")
    plt.plot(res_df["level"], res_df[metric], marker='o', label="ResNet")

    plt.title(f"{transform_type} ({metric}) - MNIST")
    plt.xlabel("Level")
    plt.ylabel(metric)
    plt.legend()

    plt.savefig(f"results/comparison/resnet_cnn_mnist_{transform_type}_{metric}.png")
    plt.close()


# =========================
# Compare All Transformations
# =========================
transforms_list = ["translation_x", "translation_y", "translation_xy", "rotation"]

for t in transforms_list:
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plot_compare(t, metric)


# =========================
# Flip Comparison
# =========================
cnn_flip = cnn[cnn["type"] == "flip"]["accuracy"].values[0]
res_flip = resnet[resnet["type"] == "flip"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "ResNet"], [cnn_flip, res_flip])
plt.title("Flip Comparison (MNIST)")
plt.ylabel("Accuracy")
plt.savefig("results/comparison/resnet_cnn_mnist_flip_accuracy.png")
plt.close()


# =========================
# Baseline Comparison
# =========================
cnn_base = cnn[cnn["type"] == "original"]["accuracy"].values[0]
res_base = resnet[resnet["type"] == "original"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "ResNet"], [cnn_base, res_base])
plt.title("Baseline Accuracy Comparison (MNIST)")
plt.ylabel("Accuracy")
plt.savefig("results/comparison/resnet_cnn_mnist_baseline_accuracy.png")
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
summary_df.to_csv("results/comparison/resnet_cnn_mnist_summary.csv", index=False)

# =========================
# Print
# =========================
print(summary_df)
print("✅ CNN vs ResNet MNIST comparison completed!")