import numpy as np
import matplotlib.pyplot as plt

baseline_hist = np.load(
    "results/baseline_history.npy",
    allow_pickle=True
).item()

mobilenet_hist = np.load(
    "results/mobilenet_history.npy",
    allow_pickle=True
).item()

# =========================
# Accuracy Comparison
# =========================
plt.figure(figsize=(6,4))
plt.plot(baseline_hist["val_accuracy"], label="Baseline Validation Accuracy")
plt.plot(mobilenet_hist["val_accuracy"], label="MobileNet Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("results/accuracy_comparison.png", dpi=200)
plt.show()
plt.close()

# =========================
# Loss Comparison
# =========================
plt.figure(figsize=(6,4))
plt.plot(baseline_hist["val_loss"], label="Baseline Validation Loss")
plt.plot(mobilenet_hist["val_loss"], label="MobileNet Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_comparison.png", dpi=200)
plt.show()
plt.close()
