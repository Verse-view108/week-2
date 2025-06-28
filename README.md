## Week 2 – Advanced Model Training & Dataset Handling

## Objective
Build and train a robust garbage classification model using **EfficientNetV2B2**, with proper handling of class imbalance, data augmentation, and performance evaluation.

---

## Dataset
- Located at: `Google Drive > MyDrive > Dataset`
- Classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- Automatically split using TensorFlow:
  - **80% Training**
  - **20% Validation**

---

## Steps Completed

### 1. Data Loading & Split
- Used `image_dataset_from_directory()` with validation split.
- Image size: `128x128`
- Batch size: `32`

### 2. Class Distribution Visualization
- Counted samples per class.
- Plotted Training, Validation, and Overall class distribution using `matplotlib`.

### 3. Class Weight Calculation
- Calculated `class_weight_dict` to handle imbalance (e.g., fewer trash samples).
- Applied during training via `model.fit()`.

### 4. Data Augmentation
- Applied `RandomFlip`, `Rotation`, `Zoom`, and `Contrast` to training images.

### 5. Model Architecture (EfficientNetV2B2)
- Loaded pretrained EfficientNetV2B2 (`imagenet`, `include_top=False`).
- Added:
  - Rescaling layer
  - Global pooling
  - Dense + Dropout
  - Output layer with `softmax`

### 6. Model Training
- Phase 1: Base model frozen; trained new top layers.
- Phase 2: Unfroze top layers of base model for fine-tuning with a smaller learning rate.
- Optimizer: `Adam`, Loss: `sparse_categorical_crossentropy`

### 7. Evaluation & Metrics
- Plotted training vs validation accuracy and loss.
- Generated full `classification_report` with precision, recall, and F1-score.

### 8. Output Saved
- `best_model.h5`: Best performing model based on validation accuracy.
- `class_indices.json`: Mapping of class indices to names.

---

## Results
- Accuracy improved gradually after applying fine-tuning.
- Class imbalance partially addressed using class weights.
- Trash class remains challenging — consider SMOTE or targeted augmentation.

---


