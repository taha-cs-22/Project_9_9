# Fish Species Recognition Using Convolutional Neural Networks

## Project Description

This project focuses on recognizing and classifying fish species from images using Convolutional Neural Networks (CNNs). The system implements two different approaches:

1. **Baseline CNN Model**: A custom CNN architecture built from scratch
2. **Transfer Learning Model**: Using MobileNet pretrained on ImageNet for improved performance

The project classifies 9 different fish species:

- Black Sea Sprat
- Gilt-Head Bream
- Horse Mackerel
- Red Mullet
- Red Sea Bream
- Sea Bass
- Shrimp
- Striped Red Mullet
- Trout

The models achieve high accuracy in fish species identification and can be deployed through a Streamlit web application for real-time inference.

---

## Live Demo (Deployed Model)

You can try the trained **MobileNet transfer learning model** directly through the Streamlit web application:

🔗 **Live Application:**
https://git-n5z9h3wsxtr6c4erlucmut.streamlit.app/

**Features of the web app:**

- Upload a fish image
- Get real-time species prediction
- View confidence score
- Display top-3 predicted classes

---

## Dataset

The project uses the **A Large Scale Fish Dataset** from Kaggle, which contains thousands of fish images across multiple species.

**Dataset Link:** https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

**Dataset Split:**

- Training set: 70%
- Validation set: 15%
- Test set: 15%

Images are resized to **128×128** and normalized before training.

---

## How to Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install tensorflow keras pillow matplotlib numpy scikit-learn seaborn streamlit
```

---

## Dataset Setup and Preprocessing

1. **Download and prepare the dataset**

   - Download the dataset from Kaggle
   - Extract the archive
   - Place the `Fish_Dataset/` folder in the project root directory
2. **Preprocess the dataset**

```bash
python code/data_pre.py
```

3. **Clean the dataset (remove corrupted images)**

```bash
python code/data_cleaning.py
```

4. **Exploratory Data Analysis (EDA)**

```bash
python code/eda.py
```

---

## Model Training

### Train Baseline CNN

```bash
python code/train_baseline.py
```

### Train MobileNet (Transfer Learning)

```bash
python code/train_mobilenet.py
```

Models are saved in the `saved_model/` directory.

---

## Model Evaluation

### Evaluate Baseline Model

```bash
python code/evaluate_baseline.py
```

### Evaluate MobileNet Model

```bash
python code/evaluation_mobilenet.py
```

Evaluation metrics:

- Accuracy
- Precision, Recall, F1-score per class
- Confusion Matrix
- Classification Report

Results are saved in the `results/` directory.

---

## Model Inference (Python)

```python
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("saved_model/mobilenet_model.h5")

image = Image.open("path/to/fish_image.jpg").convert("RGB")
image = image.resize((128, 128))
img_array = np.array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions)

CLASS_NAMES = [
    "Black Sea Sprat", "Gilt-Head Bream", "Horse Mackerel",
    "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp",
    "Striped Red Mullet", "Trout"
]

print(f"Predicted species: {CLASS_NAMES[predicted_class_idx]}")
print(f"Confidence: {np.max(predictions) * 100:.2f}%")
```

---

## Streamlit Application (Local Run)

```bash
streamlit run app.py
```

---

## Project Structure

```
├── code/
│   ├── data_pre.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── dataset.py
│   ├── model.py
│   ├── train_baseline.py
│   ├── train_mobilenet.py
│   ├── evaluate_baseline.py
│   └── evaluation_mobilenet.py
├── saved_model/
│   ├── baseline_model.h5
│   └── mobilenet_model.h5
├── results/
├── app.py
├── requirements.txt
└── README.md
```

---

## Model Performance

The **MobileNet transfer learning model** consistently outperforms the baseline CNN by leveraging pretrained ImageNet features, achieving higher accuracy and better generalization across fish species.

Detailed metrics and plots are available in the `results/` directory.
