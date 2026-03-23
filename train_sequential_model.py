import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_multimodal_model
from tensorflow.keras.callbacks import EarlyStopping

# Load sequential features
FEATURES_DIR = "dataset/features"
audio_feats = np.load(os.path.join(FEATURES_DIR, "sequential_audio.npy"))  # (100,160,13)
motion_feats = np.load(os.path.join(FEATURES_DIR, "sequential_motion.npy"))  # (100,150,18)
beh_feats = np.load(os.path.join(FEATURES_DIR, "sequential_beh.npy"))  # (100,3)
labels = np.load(os.path.join(FEATURES_DIR, "labels.npy"))  # (100,) binary 0/1

print(f"Loaded data:")
print(f"  Audio shape: {audio_feats.shape}")
print(f"  Motion shape: {motion_feats.shape}")
print(f"  Beh shape: {beh_feats.shape}")
print(f"  Labels: {labels.shape}, ASD ratio: {np.mean(labels):.2f}")

# Train/test split
X_audio_train, X_audio_test, X_motion_train, X_motion_test, X_beh_train, X_beh_test, y_train, y_test = train_test_split(
    audio_feats, motion_feats, beh_feats, labels, test_size=0.2, random_state=42, stratify=labels
)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Build model
model = build_multimodal_model(mfcc_time_steps=160, mfcc_dim=13, video_frames=150, video_feat_dim=18, behav_dim=3)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    [X_audio_train, X_motion_train, X_beh_train],
    y_train,
    validation_data=([X_audio_test, X_motion_test, X_beh_test], y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
test_loss, test_acc, test_auc = model.evaluate([X_audio_test, X_motion_test, X_beh_test], y_test)
print(f"\n✅ Test Accuracy: {test_acc:.3f}")
print(f"  Test AUC: {test_auc:.3f}")

# Save
model.save("autism_model.keras")
print("✅ Improved model saved as autism_model.keras - app.py ready!")
print("Run app.py to test improved accuracy.")
