import numpy as np
import matplotlib.pyplot as plt

# Load MobileNet history
mobilenet_hist = np.load(
    "results/mobilenet_history.npy",
    allow_pickle=True
).item()

# =========================
# Figure: MobileNet Accuracy
# =========================
plt.figure(figsize=(6,4))
plt.plot(mobilenet_hist["accuracy"], label="Train Accuracy")
plt.plot(mobilenet_hist["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNet Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/mobilenet_accuracy_curve.png", dpi=200)
plt.show()
plt.close()

# =========================
# Figure: MobileNet Loss
# =========================
plt.figure(figsize=(6,4))
plt.plot(mobilenet_hist["loss"], label="Train Loss")
plt.plot(mobilenet_hist["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNet Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/mobilenet_loss_curve.png", dpi=200)
plt.show()
plt.close()
