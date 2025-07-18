# -*- coding: utf-8 -*-
"""Week2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16AaoxbRVglbJ_jxyepd2hRcLuevz1vyJ
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Dataset path
dataset_dir = '/content/drive/MyDrive/Dataset'

# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import classification_report

# Load datasets
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)

# Class names
class_names = train_ds_raw.class_names
print("Class names:", class_names)

# Save class indices
with open("class_indices.json", "w") as f:
    json.dump({i: name for i, name in enumerate(class_names)}, f)

# Class distribution
def count_distribution(dataset, class_names):
    total = 0
    counts = {name: 0 for name in class_names}
    for _, labels in dataset.unbatch():
        counts[class_names[labels.numpy()]] += 1
        total += 1
    for k in counts:
        counts[k] = round((counts[k] / total) * 100, 2)
    return counts

train_dist = count_distribution(train_ds_raw, class_names)
val_dist = count_distribution(val_ds, class_names)
overall_dist = {k: round((train_dist[k] + val_dist[k]) / 2, 2) for k in train_dist}

print("train_dist:", train_dist)
print("val_dist:", val_dist)
print("overall_dist:", overall_dist)

def simple_bar_plot_dist(title, dist):
    plt.bar(class_names, list(dist.values()), color='skyblue')
    plt.title(title)
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

simple_bar_plot_dist('Training Set Class Distribution (%)', train_dist)
simple_bar_plot_dist('Validation Set Class Distribution (%)', val_dist)
simple_bar_plot_dist('Overall Class Distribution (%)', overall_dist)

# Class weights
class_counts = {i: 0 for i in range(len(class_names))}
for _, labels in train_ds_raw.unbatch():
    class_counts[labels.numpy()] += 1

total_samples = sum(class_counts.values())
n_classes = len(class_names)
class_weight_dict = {i: (total_samples / (n_classes * count)) for i, count in class_counts.items()}
print("Class weights:", class_weight_dict)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Dataset pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Show some training samples
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
plt.show()

# Base Model with preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B2, preprocess_input

base_model = EfficientNetV2B2(
    include_top=False,
    weights='imagenet',
    input_shape=(128, 128, 3),
    pooling='max'
)
base_model.trainable = False  # Freeze base initially

# Improved Top Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Lambda(preprocess_input),
    base_model,
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)

# Train - Phase 1 (Frozen base)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, model_checkpoint]
)

# Unfreeze some top layers of base model (fine-tuning)
base_model.trainable = True
for layer in base_model.layers[:-40]:  # Keep most layers frozen
    layer.trainable = False

# Compile again with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune - Phase 2 (Unfrozen top layers)
fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, model_checkpoint]
)

# Plot results
def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

plot_metrics(history)
plot_metrics(fine_tune_history)

# Final evaluation
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))