import numpy as np
import matplotlib.pyplot as plt

history = np.load("results/baseline_history.npy", allow_pickle=True).item()

# ===== Accuracy Curve =====
plt.figure(figsize=(6,4))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/baseline_accuracy_curve.png", dpi=200)
plt.show()
plt.close()

# ===== Loss Curve =====
plt.figure(figsize=(6,4))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/baseline_loss_curve.png", dpi=200)
plt.show()
plt.close()
