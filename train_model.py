import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ================= LOAD DATA =================
AUDIO_FEATURES_PATH = "dataset/features/audio_features.npy"
MOTION_FEATURES_PATH = "dataset/features/motion_features.npy"
LABELS_PATH = "dataset/features/labels.npy"

audio_features = np.load(AUDIO_FEATURES_PATH)
motion_features = np.load(MOTION_FEATURES_PATH)
labels = np.load(LABELS_PATH)

print("Audio features shape:", audio_features.shape)
print("Motion features shape:", motion_features.shape)
print("Labels shape:", labels.shape)

# ================= ENCODE LABELS =================
le = LabelEncoder()
y = le.fit_transform(labels)  # Converts string labels to integers

# ================= TRAIN/TEST SPLIT =================
X_audio_train, X_audio_test, X_motion_train, X_motion_test, y_train, y_test = train_test_split(
    audio_features, motion_features, y, test_size=0.2, random_state=42
)

# ================= BUILD MODEL =================
def build_multimodal_model(audio_dim, motion_dim):
    # Audio branch
    audio_input = Input(shape=(audio_dim,), name="audio_input")
    x1 = layers.Dense(64, activation="relu")(audio_input)
    x1 = layers.Dropout(0.3)(x1)

    # Motion branch
    motion_input = Input(shape=(motion_dim,), name="motion_input")
    x2 = layers.Dense(64, activation="relu")(motion_input)
    x2 = layers.Dropout(0.3)(x2)

    # Merge
    merged = layers.concatenate([x1, x2])
    x = layers.Dense(64, activation="relu")(merged)
    output = layers.Dense(len(np.unique(y)), activation="softmax")(x)

    model = Model(inputs=[audio_input, motion_input], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_multimodal_model(audio_features.shape[1], motion_features.shape[1])
model.summary()

# ================= TRAIN =================
model.fit(
    [X_audio_train, X_motion_train],
    y_train,
    validation_data=([X_audio_test, X_motion_test], y_test),
    epochs=10,
    batch_size=8,
    verbose=1
)

# ================= SAVE MODEL =================
model.save("autism_audio_motion_model.keras")
print("✅ Model saved as autism_audio_motion_model.keras")
