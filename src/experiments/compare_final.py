import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# File Paths
# =========================
cnn_path = "results/metrics/invariance_full_cnn.csv"
cbam_path = "results/metrics/invariance_full_cbam.csv"

# =========================
# Check Files Exist
# =========================
if not os.path.exists(cnn_path):
    raise FileNotFoundError(f"{cnn_path} not found")

if not os.path.exists(cbam_path):
    raise FileNotFoundError(f"{cbam_path} not found")

# =========================
# Load Data
# =========================
cnn = pd.read_csv(cnn_path)
cbam = pd.read_csv(cbam_path)

print("CNN loaded:", cnn.shape)
print("CBAM loaded:", cbam.shape)

# =========================
# Create Output Folder
# =========================
os.makedirs("results/comparison", exist_ok=True)

# =========================
# Plot Function
# =========================
def plot_compare(transform, metric):
    cnn_df = cnn[cnn["type"] == transform]
    cbam_df = cbam[cbam["type"] == transform]

    if cnn_df.empty or cbam_df.empty:
        print(f"Skipping {transform} ({metric}) — no data")
        return

    plt.figure()
    plt.plot(cnn_df["level"], cnn_df[metric], marker='o', label="CNN")
    plt.plot(cbam_df["level"], cbam_df[metric], marker='o', label="CBAM")

    plt.title(f"{transform} - {metric}")
    plt.xlabel("Level")
    plt.ylabel(metric)
    plt.legend()

    filename = f"results/comparison/{transform}_{metric}.png"
    plt.savefig(filename)
    plt.close()

    print(f"Saved: {filename}")

# =========================
# Generate Plots
# =========================
transforms_list = ["translation_x", "translation_y", "translation_xy", "rotation"]
metrics = ["accuracy", "precision", "recall", "f1"]

for t in transforms_list:
    for m in metrics:
        plot_compare(t, m)

# =========================
# Flip Comparison
# =========================
cnn_flip = cnn[cnn["type"] == "flip"]["accuracy"].values[0]
cbam_flip = cbam[cbam["type"] == "flip"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "CBAM"], [cnn_flip, cbam_flip])
plt.title("Flip Accuracy Comparison")
plt.savefig("results/comparison/flip_accuracy.png")
plt.close()

# =========================
# Baseline Comparison
# =========================
cnn_base = cnn[cnn["type"] == "original"]["accuracy"].values[0]
cbam_base = cbam[cbam["type"] == "original"]["accuracy"].values[0]

plt.figure()
plt.bar(["CNN", "CBAM"], [cnn_base, cbam_base])
plt.title("Baseline Accuracy")
plt.savefig("results/comparison/baseline_accuracy.png")
plt.close()

# =========================
# Summary Table
# =========================
summary = []

for t in ["original", "translation_x", "translation_y", "translation_xy", "rotation", "flip"]:
    cnn_val = cnn[cnn["type"] == t]["accuracy"].mean()
    cbam_val = cbam[cbam["type"] == t]["accuracy"].mean()

    summary.append({
        "type": t,
        "cnn_accuracy": cnn_val,
        "cbam_accuracy": cbam_val
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/comparison/summary.csv", index=False)

print("\n📊 Summary:")
print(summary_df)

print("\n✅ Comparison completed successfully!")