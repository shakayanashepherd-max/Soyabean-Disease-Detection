
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR, "soybean_rust_mobilenet.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("       LEAFLENSAI - SOYBEAN RUST MODEL TRAINING")
print("=" * 60)


# ==========================================================
# 2. CHECK DATASET FOLDERS
# ==========================================================

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        "Dataset folder not found. Create dataset/healthy "
        "and dataset/rust inside your project folder."
    )

for folder in ["healthy", "rust"]:
    path = os.path.join(DATASET_DIR, folder)

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing folder: {path}"
        )


# ==========================================================
# 3. DATA AUGMENTATION AND VALIDATION SPLIT
# ==========================================================

# Augmentation is applied only to training images.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20,
    rotation_range=25,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest"
)

# Validation images are not augmented.
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED,
    classes=["healthy", "rust"]
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED,
    classes=["healthy", "rust"]
)

print("\nClass mapping:", train_generator.class_indices)
print("Training images:", train_generator.samples)
print("Validation images:", validation_generator.samples)

if train_generator.class_indices != {"healthy": 0, "rust": 1}:
    raise ValueError(
        "Incorrect class mapping. Expected healthy=0 and rust=1."
    )

if train_generator.samples == 0 or validation_generator.samples == 0:
    raise ValueError("No images found. Check your dataset folders.")


# ==========================================================
# 4. BALANCE THE CLASSES
# ==========================================================

class_ids = np.unique(train_generator.classes)

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=train_generator.classes
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(class_ids, weights)
}

print("\nClass weights:", class_weights)


# ==========================================================
# 5. BUILD MOBILENETV2 MODEL
# ==========================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Stage 1: freeze the pretrained feature extractor.
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

features = base_model(inputs, training=False)
features = GlobalAveragePooling2D()(features)
features = BatchNormalization()(features)
features = Dropout(0.30)(features)
features = Dense(128, activation="relu")(features)
features = Dropout(0.20)(features)

# Output: probability of rust.
outputs = Dense(1, activation="sigmoid")(features)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# ==========================================================
# 6. TRAINING CALLBACKS
# ==========================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr
]


# ==========================================================
# 7. STAGE 1: TRAIN CLASSIFICATION HEAD
# ==========================================================

print("\n" + "=" * 60)
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 60)

history1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ==========================================================
# 8. STAGE 2: FINE-TUNE MOBILENETV2
# ==========================================================

print("\n" + "=" * 60)
print("STAGE 2: FINE-TUNING MOBILENETV2")
print("=" * 60)

# Unfreeze the later layers so they can adapt to soybean leaves.
base_model.trainable = True

# Keep earlier layers frozen; fine-tune the last 30 layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Batch normalization layers are kept frozen for stability.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# Save a checkpoint only when validation loss improves.
# Reset callback state for this training stage.
checkpoint2 = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping2 = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-8,
    verbose=1
)

history2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    initial_epoch=len(history1.epoch),
    epochs=len(history1.epoch) + FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint2,
        early_stopping2,
        reduce_lr2
    ]
)


# ==========================================================
# 9. LOAD BEST SAVED MODEL
# ==========================================================

print("\nLoading best saved model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Best model loaded from:", MODEL_PATH)


# ==========================================================
# 10. EVALUATE MODEL
# ==========================================================

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

validation_generator.reset()

results = model.evaluate(
    validation_generator,
    verbose=1,
    return_dict=True
)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

validation_generator.reset()

probabilities = model.predict(
    validation_generator,
    verbose=1
).ravel()

# Default classification threshold: 0.50
predicted_classes = (probabilities >= 0.50).astype(int)
actual_classes = validation_generator.classes

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        actual_classes,
        predicted_classes,
        labels=[0, 1],
        target_names=["Healthy", "Rust"],
        digits=4,
        zero_division=0
    )
)

cm = confusion_matrix(
    actual_classes,
    predicted_classes,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print(cm)

# Save confusion matrix.
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy", "Rust"]
)

display.plot(values_format="d")
plt.title("LeafLensAI - Soybean Rust Detection")
plt.tight_layout()
plt.savefig(
    os.path.join(MODEL_DIR, "confusion_matrix.png"),
    dpi=200
)
plt.close()


# ==========================================================
# 11. RUST-SPECIFIC METRICS
# ==========================================================

true_healthy = cm[0, 0]
false_rust = cm[0, 1]
missed_rust = cm[1, 0]
detected_rust = cm[1, 1]

rust_precision = (
    detected_rust / (detected_rust + false_rust)
    if detected_rust + false_rust > 0 else 0
)

rust_recall = (
    detected_rust / (detected_rust + missed_rust)
    if detected_rust + missed_rust > 0 else 0
)

rust_f1 = (
    2 * rust_precision * rust_recall /
    (rust_precision + rust_recall)
    if rust_precision + rust_recall > 0 else 0
)

print("\n" + "=" * 60)
print("RUST DETECTION PERFORMANCE")
print("=" * 60)

print(f"Rust Precision: {rust_precision:.2%}")
print(f"Rust Recall:    {rust_recall:.2%}")
print(f"Rust F1-Score:  {rust_f1:.2%}")
print(f"Rust cases detected: {detected_rust}")
print(f"Rust cases missed:   {missed_rust}")
print(f"Healthy leaves incorrectly flagged as rust: {false_rust}")


# ==========================================================
# 12. SAVE TRAINING GRAPHS
# ==========================================================

def combine_history(histories, metric):
    values = []
    validation_values = []

    for history in histories:
        values.extend(history.history.get(metric, []))
        validation_values.extend(
            history.history.get("val_" + metric, [])
        )

    return values, validation_values


accuracy, val_accuracy = combine_history(
    [history1, history2], "accuracy"
)

loss, val_loss = combine_history(
    [history1, history2], "loss"
)

plt.figure(figsize=(10, 5))
plt.plot(accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.title("LeafLensAI Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(MODEL_DIR, "training_accuracy.png"),
    dpi=200
)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("LeafLensAI Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(MODEL_DIR, "training_loss.png"),
    dpi=200
)
plt.close()


# ==========================================================
# 13. FINISH
# ==========================================================

print("\n" + "=" * 60)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 60)

print("Saved model:", MODEL_PATH)
print("Saved confusion matrix: models/confusion_matrix.png")
print("Saved accuracy graph: models/training_accuracy.png")
print("Saved loss graph: models/training_loss.png")

print(
    "\nRemember: validation performance is not a guarantee "
    "of accuracy on new real-world images."
)
