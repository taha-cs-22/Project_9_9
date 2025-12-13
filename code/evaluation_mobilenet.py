from dataset import create_generators
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_gen, val_gen, test_gen = create_generators("dataset_ready")

model = load_model("saved_model/mobilenet_model.h5")

y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

print(classification_report(
    y_true,
    y_pred,
    target_names=list(test_gen.class_indices.keys())
))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=test_gen.class_indices.keys(),
            yticklabels=test_gen.class_indices.keys())
plt.show()
