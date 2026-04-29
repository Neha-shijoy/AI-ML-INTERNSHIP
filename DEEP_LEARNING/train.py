


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import json

# ──────────────────────────────────────────────
# 1.  CLASS LABELS
# ──────────────────────────────────────────────
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ──────────────────────────────────────────────
# 2.  LOAD & PRE-PROCESS DATA
# ──────────────────────────────────────────────
def load_and_preprocess():
    print("📦 Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test,  10)

    print(f"   Train samples : {x_train.shape[0]:,}")
    print(f"   Test  samples : {x_test.shape[0]:,}")
    print(f"   Image shape   : {x_train.shape[1:]}")
    return x_train, y_train_cat, x_test, y_test_cat


# ──────────────────────────────────────────────
# 3.  BUILD THE CNN ARCHITECTURE
# ──────────────────────────────────────────────
def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Architecture overview
    ─────────────────────
    Block 1  : Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout(0.25)
    Block 2  : Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout(0.25)
    Block 3  : Conv(128)→ BN → ReLU → Conv(128)→ BN → ReLU → MaxPool → Dropout(0.25)
    Head     : Flatten → Dense(512) → BN → ReLU → Dropout(0.5) → Dense(10, softmax)
    """
    model = models.Sequential([
        # ── Block 1 ──────────────────────────────
        layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 2 ──────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 3 ──────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Classification Head ───────────────────
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="CIFAR10_CNN")

    return model


# ──────────────────────────────────────────────
# 4.  DATA AUGMENTATION
# ──────────────────────────────────────────────
def get_augmenter():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )


# ──────────────────────────────────────────────
# 5.  TRAIN
# ──────────────────────────────────────────────
def train(model, x_train, y_train, x_test, y_test, epochs=30, batch_size=64):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint("cnn_best.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    aug = get_augmenter()
    aug.fit(x_train)

    print("\n🚀 Starting training …")
    history = model.fit(
        aug.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────
# 6.  PLOT TRAINING CURVES
# ──────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training History – CIFAR-10", fontsize=15, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Acc", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",   linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("📊 Training curves saved → training_curves.png")


# ──────────────────────────────────────────────
# 7.  EVALUATE & SAVE
# ──────────────────────────────────────────────
def evaluate_and_save(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy : {acc * 100:.2f}%")
    print(f"   Test Loss     : {loss:.4f}")

    # Save final model
    model.save("cnn_cifar10.h5")
    print("💾 Model saved → cnn_cifar10.h5")

    # Save class names for Streamlit
    with open("class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f)
    print("💾 Class names saved → class_names.json")

    return acc


# ──────────────────────────────────────────────
# 8.  MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    x_train, y_train, x_test, y_test = load_and_preprocess()
    model = build_cnn()
    history = train(model, x_train, y_train, x_test, y_test, epochs=30)
    plot_history(history)
    evaluate_and_save(model, x_test, y_test)

    print("\n🎉 Done!  Run the Streamlit app with:")
    print("   streamlit run app.py")