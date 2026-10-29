
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Dataset and model paths
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            "Dataset folder nahi mila. Project folder check karo."
        )

    # Load images arranged in class folders
    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print("Plant disease classes:", class_names)

    # Improve data loading performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # CNN model
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train the model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save model and class names
    model.save(MODEL_DIR / "plant_disease_model.keras")

    with open(MODEL_DIR / "class_names.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(class_names))

    print("\nTraining complete!")
    print("Model saved in:", MODEL_DIR / "plant_disease_model.keras")
    print("Class names saved in:", MODEL_DIR / "class_names.txt")

if __name__ == "__main__":
    main()
