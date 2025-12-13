# Fish Species Recognition Using Convolutional Neural Networks

## Overview
This project focuses on recognizing and classifying fish species from images using a Convolutional Neural Network (CNN). The goal is to build an image classification model capable of identifying different fish species based on visual characteristics such as shape, color patterns, fins, and scales.

The project serves as an applied example of deep learning in computer vision and demonstrates how CNNs can be used in real-world ecological and commercial applications.

## Motivation and Significance
Automated fish species recognition has practical importance in several domains, including:

- Monitoring marine biodiversity
- Supporting fisheries management and sustainability
- Assisting ecological and environmental research
- Helping conservation programs track vulnerable or overfished species
- Providing educational tools for marine biology and ecology

By automating the identification process, this approach reduces manual effort and increases accuracy and scalability.

## Applications
Potential applications of this project include:

- Marine biodiversity monitoring systems
- Automated fish sorting and quality control in fish markets
- Educational software for species identification
- Conservation and environmental monitoring programs
- Research on fish populations and habitats

## Dataset
The model can be trained using publicly available fish image datasets, such as:

- Fish species classification datasets from Kaggle
- Academic datasets from marine biology research
- Custom datasets collected from fish markets or underwater cameras

A typical dataset split is:

- **Training set:** 70%
- **Validation set:** 15%
- **Test set:** 15%

## Methodology

### Model Architecture
The project uses a CNN designed to capture fine-grained visual features. A typical architecture includes:

- Convolutional layers with ReLU activation
- Max pooling layers for spatial reduction
- Fully connected (dense) layers for classification
- Dropout layers to reduce overfitting
- A Softmax output layer for multi-class classification

The network is implemented using **TensorFlow/Keras**.

### Data Preprocessing
Before training, images undergo several preprocessing steps:

- Resizing to a fixed resolution (e.g., 128 × 128)
- Normalizing pixel values to the range [0, 1]
- Data augmentation to improve generalization, including:
  - Rotation and horizontal flipping
  - Zooming and shifting
  - Brightness and contrast adjustments

### Training
Key training settings include:

- **Optimizer:** Adam
- **Loss function:** Categorical Cross-Entropy
- **Batch size:** 32–64
- **Number of epochs:** 20–40
- Early stopping to prevent overfitting
- Model checkpointing to save the best model

### Evaluation
Model performance is evaluated using:

- Overall classification accuracy
- Precision, recall, and F1-score for each species
- Confusion matrix to analyze misclassifications

These metrics help identify strengths and weaknesses of the model, especially for visually similar species.

## Challenges and Future Improvements

**Common challenges include:**

- High visual similarity between certain fish species
- Variations in lighting, background, and image quality
- Partial occlusion in underwater or market images

**Possible improvements:**

- Applying transfer learning using pretrained models such as ResNet, MobileNet, or EfficientNet
- Increasing dataset size and diversity
- Performing fish segmentation before classification
- Hyperparameter tuning and deeper architectures

## Conclusion
This project demonstrates the effectiveness of convolutional neural networks in classifying fish species from images. With improved datasets and more advanced models, the system can be extended for real-world deployment in ecological monitoring, fisheries management, and market automation.

## References
- Fish species classification datasets (Kaggle and academic sources)
- Research papers on CNN-based fish recognition
- TensorFlow and Keras official documentation

## Instructions on How to Run the Code
1. **Install required packages**  
- pip install tensorflow keras pillow matplotlib numpy

2. **Prepare the dataset**  
- Download the dataset from [Kaggle: A Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset).  
- Extract the downloaded archive. The folder `Fish_Dataset/` will be included in the extracted files.  
- Place the `Fish_Dataset/` folder into the root of the project.
  
3. **Preprocess the dataset**  
- Run the preprocessing script to generate the structured dataset in `dataset_ready/`:
- python code/data_pre.py
    
4. **Cleaning / organization**  
- Run the data cleaning script :
- python code/data_cleaning.py
  
5. **Explore the dataset**  
- Run the exploratory data analysis script to view image statistics and sample images:
- python code/eda.py
  
6. **Evaluate the baselin model**  
- Evaluate the baseline model (baseline_model.h5) on the test set to get performance metrics:
- python code/evaluate_baseline.py
- This will output accuracy, precision, recall, F1-score, and a confusion matrix.
  
7. **Evaluate the transfer learning (MobileNet)**  
- Evaluate the MobileNet model (mobilenet_model.h5) on the test set to get performance metrics:
- python code/evaluation_mobilenet.py
- This will output accuracy, precision, recall, F1-score, and a confusion matrix.

 
