from dataset import create_generators
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

_, _, test_gen = create_generators("dataset_ready")
model = load_model("saved_model/baseline_model.h5")

y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

# save classification report
report = classification_report(
    y_true, y_pred,
    target_names=list(test_gen.class_indices.keys()),
    digits=4
)
with open("results/baseline_classification_report.txt", "w") as f:
    f.write(report)

# save confusion matrix image WITH numbers
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Baseline Confusion Matrix")
plt.colorbar()

ticks = np.arange(len(test_gen.class_indices))
labels = list(test_gen.class_indices.keys())

plt.xticks(ticks, labels, rotation=45, ha="right")
plt.yticks(ticks, labels)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=10
        )

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/baseline_confusion_matrix.png", dpi=200)
plt.close()

# normalized confusion matrix WITH numbers
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]

plt.figure(figsize=(8,6))
plt.imshow(cm_norm)
plt.title("Normalized Confusion Matrix")
plt.colorbar()

plt.xticks(ticks, labels, rotation=45, ha="right")
plt.yticks(ticks, labels)

for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(
            j, i, f"{cm_norm[i, j]:.2f}",
            ha="center", va="center",
            color="white" if cm_norm[i, j] > 0.5 else "black",
            fontsize=10
        )

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/baseline_confusion_matrix_normalized.png", dpi=200)
plt.close()


print("Saved: results/baseline_classification_report.txt")
print("Saved: results/baseline_confusion_matrix.png")
print("Saved: results/baseline_confusion_matrix_normalized.png")