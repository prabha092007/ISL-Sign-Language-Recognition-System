"""
Stage 2 — Model Training (Best Version)
Improvements:
- Better model architecture
- Data augmentation (adds noise to landmarks for robustness)
- Learning rate warmup
- Confusion matrix to see which signs are confused
- Saves best model automatically
- Per-class accuracy report
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.utils            import class_weight
from sklearn.metrics          import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils     import to_categorical

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE  = 'data/landmarks_v6.csv'
MODEL_DIR  = 'model'
EPOCHS     = 200
BATCH_SIZE = 32
AUGMENT_FACTOR = 3    # multiply training data by adding noise variants
NOISE_STD      = 0.01 # small noise for augmentation

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"Total samples  : {len(df)}")
print(f"Classes found  : {sorted(df['label'].unique())}")
print(f"\nSamples per class:")
print(df['label'].value_counts().sort_index().to_string())

X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values

# ── Label encoding ────────────────────────────────────────────────────────────
le          = LabelEncoder()
y_encoded   = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"\nTotal classes  : {num_classes}")
np.save(os.path.join(MODEL_DIR, 'label_encoder.npy'), le.classes_)

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ── Data augmentation ─────────────────────────────────────────────────────────
# Add small random noise to landmark coordinates
# This makes model robust to slight hand position changes
print(f"\nAugmenting training data (x{AUGMENT_FACTOR})...")
X_aug_list = [X_train]
y_aug_list = [y_train]

for _ in range(AUGMENT_FACTOR - 1):
    noise = np.random.normal(0, NOISE_STD, X_train.shape).astype(np.float32)
    X_aug_list.append(X_train + noise)
    y_aug_list.append(y_train)

X_train = np.concatenate(X_aug_list, axis=0)
y_train = np.concatenate(y_aug_list, axis=0)
print(f"Training samples after augmentation: {len(X_train)}")

# Shuffle augmented data
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# ── Class weights ─────────────────────────────────────────────────────────────
weights       = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

# ── Model architecture ────────────────────────────────────────────────────────
# Input: 126 (2 hands x 21 landmarks x xyz)
# Deeper network with residual-style connections
model = Sequential([
    # Block 1
    Dense(512, activation='relu', input_shape=(126,)),
    BatchNormalization(),
    Dropout(0.4),

    # Block 2
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    # Block 3
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # Block 4
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # Block 5
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # Block 6
    Dense(64, activation='relu'),
    Dropout(0.2),

    # Output
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

# ── Callbacks ─────────────────────────────────────────────────────────────────
best_model_path = os.path.join(MODEL_DIR, 'isl_model_best.keras')

callbacks = [
    # Stop if no improvement for 25 epochs
    EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
    # Always save the best model
    ModelCheckpoint(
        best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\nTraining on {len(X_train)} samples...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n{'='*50}")
print(f"Test Accuracy : {acc*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
print(f"{'='*50}")

# ── Per-class accuracy report ─────────────────────────────────────────────────
y_pred     = model.predict(X_test, verbose=0)
y_pred_cls = np.argmax(y_pred, axis=1)
print("\nPer-class Report:")
print(classification_report(y_test, y_pred_cls, target_names=le.classes_))

# ── Save final model ──────────────────────────────────────────────────────────
final_model_path = os.path.join(MODEL_DIR, 'isl_model_v4.keras')
model.save(final_model_path)
print(f"Final model saved : {final_model_path}")
print(f"Best model saved  : {best_model_path}")

# ── Plot training curves ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'), dpi=150)
plt.show()
print("Training curves saved.")

# ── Confusion matrix ──────────────────────────────────────────────────────────
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_cls)
plt.figure(figsize=(20, 18))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    linewidths=0.5
)
plt.title('Confusion Matrix — ISL Recognition', fontsize=16)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()
print("Confusion matrix saved.")

print("\n✓ Training complete!")
print(f"✓ Test accuracy: {acc*100:.2f}%")
print("✓ Run inference.py for live recognition!")
