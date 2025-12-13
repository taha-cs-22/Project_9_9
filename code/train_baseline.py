from dataset import create_generators
from model import build_baseline_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

IMG_SIZE = (128, 128)
EPOCHS = 30

train_gen, val_gen, test_gen = create_generators("dataset_ready")
num_classes = train_gen.num_classes

model = build_baseline_model(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=num_classes
)

model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(
        "saved_model/baseline_model.h5",
        save_best_only=True
    )
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

os.makedirs("results", exist_ok=True)

np.save("results/baseline_history.npy", history.history)

