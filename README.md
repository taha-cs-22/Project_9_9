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

## Dataset
The project uses the **A Large Scale Fish Dataset** from Kaggle, which contains thousands of fish images across multiple species.

**Dataset Link**: [Kaggle: A Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)

**Dataset Structure**:
- **Training set**: 70%
- **Validation set**: 15% 
- **Test set**: 15%

Images are preprocessed to 128×128 pixels and normalized for training.

## How to Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

Or install individual packages:

```bash
pip install tensorflow keras pillow matplotlib numpy scikit-learn seaborn streamlit
```

## Dataset Setup and Preprocessing

1. **Download and prepare the dataset**:
   - Download the dataset from the Kaggle link above
   - Extract the downloaded archive
   - Place the `Fish_Dataset/` folder in the project root directory

2. **Preprocess the dataset**:
   ```bash
   python code/data_pre.py
   ```

3. **Clean the dataset** (remove corrupted images):
   ```bash
   python code/data_cleaning.py
   ```

4. **Explore the dataset** (generate visualizations and statistics):
   ```bash
   python code/eda.py
   ```

## How to Run Training Scripts

### Train Baseline Model
```bash
python code/train_baseline.py
```

### Train MobileNet Transfer Learning Model  
```bash
python code/train_mobilenet.py
```

### Train Both Models (Alternative)
```bash
python code/train.py
```

Training will:
- Use early stopping to prevent overfitting
- Save the best model weights automatically
- Generate training history for analysis
- Save models to `saved_model/` directory

## How to Run Evaluation

### Evaluate Baseline Model
```bash
python code/evaluate_baseline.py
```

### Evaluate MobileNet Model
```bash
python code/evaluation_mobilenet.py
```

Evaluation outputs:
- Classification accuracy
- Precision, recall, and F1-score per class
- Confusion matrix visualization
- Detailed classification report

## How to Load Saved Models for Inference

### Using Python Script
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("saved_model/mobilenet_model.h5")

# Load and preprocess an image
image = Image.open("path/to/fish_image.jpg").convert("RGB")
image = image.resize((128, 128))
img_array = np.array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions)

# Class names
CLASS_NAMES = [
    "Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel",
    "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp",
    "Striped Red Mullet", "Trout"
]

print(f"Predicted species: {CLASS_NAMES[predicted_class_idx]}")
print(f"Confidence: {np.max(predictions)*100:.2f}%")
```

### Using Streamlit Web Application
```bash
streamlit run app.py
```

This launches a web interface where you can:
- Upload fish images
- Get real-time predictions
- View confidence scores
- See top-3 predictions

## Project Structure
```
├── code/
│   ├── data_pre.py          # Dataset preprocessing
│   ├── data_cleaning.py     # Remove corrupted images
│   ├── eda.py              # Exploratory data analysis
│   ├── dataset.py          # Data generators
│   ├── model.py            # Model architectures
│   ├── train_baseline.py   # Train baseline CNN
│   ├── train_mobilenet.py  # Train MobileNet model
│   ├── evaluate_baseline.py    # Evaluate baseline model
│   └── evaluation_mobilenet.py # Evaluate MobileNet model
├── saved_model/
│   ├── baseline_model.h5   # Trained baseline model
│   └── mobilenet_model.h5  # Trained MobileNet model
├── results/                # Training results and plots
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Model Performance
The MobileNet transfer learning model typically achieves higher accuracy compared to the baseline CNN due to leveraging pretrained ImageNet features. Detailed performance metrics are generated during evaluation and saved in the `results/` directory.
