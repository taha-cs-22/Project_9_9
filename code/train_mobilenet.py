from dataset import create_generators
from model import build_mobilenet_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

IMG_SIZE = (128,128)
EPOCHS = 30

train_gen, val_gen, _ = create_generators("dataset_ready")
num_classes = train_gen.num_classes

model = build_mobilenet_model((IMG_SIZE[0], IMG_SIZE[1], 3), num_classes)

model.compile(
    optimizer=Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ModelCheckpoint("saved_model/mobilenet_model.h5", save_best_only=True)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

os.makedirs("results", exist_ok=True)
np.save("results/mobilenet_history.npy", history.history)