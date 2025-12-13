from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.applications import MobileNet

#Baseline

def build_baseline_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),

        Dense(num_classes, activation="softmax")
    ])
    return model

#MobileNet

def build_mobilenet_model(input_shape, num_classes):
    base_model = MobileNet(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    return model
