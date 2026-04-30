import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Load Results
# =========================
cnn = pd.read_csv("results/metrics/invariance_cifar_cnn.csv")
cbam = pd.read_csv("results/metrics/invariance_cifar_cbam.csv")

# =========================
# Create Output Folder
# =========================
os.makedirs("results/plots", exist_ok=True)

# =========================
# Plot Single Model
# =========================
def plot_single(df, transform, model_name):
    sub = df[df["type"] == transform]

    if sub.empty:
        return

    plt.figure()
    plt.plot(sub["level"], sub["accuracy"], marker='o')
    plt.title(f"{model_name.upper()} - {transform} (CIFAR-10)")
    plt.xlabel("Level")
    plt.ylabel("Accuracy")

    plt.savefig(f"results/plots/cifar_{model_name}_{transform}.png")
    plt.close()

# =========================
# Plot Comparison
# =========================
def plot_compare(transform):
    cnn_df = cnn[cnn["type"] == transform]
    cbam_df = cbam[cbam["type"] == transform]

    if cnn_df.empty or cbam_df.empty:
        return

    plt.figure()
    plt.plot(cnn_df["level"], cnn_df["accuracy"], marker='o', label="CNN")
    plt.plot(cbam_df["level"], cbam_df["accuracy"], marker='o', label="CBAM")

    plt.title(f"CIFAR-10 Comparison - {transform}")
    plt.xlabel("Level")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(f"results/plots/cifar_compare_{transform}.png")
    plt.close()

# =========================
# Transform List
# =========================
transforms_list = [
    "translation_x",
    "translation_y",
    "translation_xy",
    "rotation"
]

# =========================
# Generate Individual Plots
# =========================
for t in transforms_list:
    plot_single(cnn, t, "cnn")
    plot_single(cbam, t, "cbam")

# =========================
# Generate Comparison Plots
# =========================
for t in transforms_list:
    plot_compare(t)

# =========================
# Flip Comparison
# =========================
cnn_flip = cnn[cnn["type"] == "flip"]["accuracy"].values[0]
cbam_flip = cbam[cbam["type"] == "flip"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "CBAM"], [cnn_flip, cbam_flip])
plt.title("CIFAR-10 Flip Comparison")
plt.ylabel("Accuracy")

plt.savefig("results/plots/cifar_compare_flip.png")
plt.close()

# =========================
# Baseline Comparison
# =========================
cnn_base = cnn[cnn["type"] == "original"]["accuracy"].values[0]
cbam_base = cbam[cbam["type"] == "original"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "CBAM"], [cnn_base, cbam_base])
plt.title("CIFAR-10 Baseline Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("results/plots/cifar_compare_baseline.png")
plt.close()

# =========================
# Done
# =========================
print("✅ All CIFAR plots generated successfully!")