import os
import json
import numpy as np
import librosa
import pandas as pd

# ================= CONFIGURATION =================
DATASET_DIR = "dataset"
VOICE_DIR = os.path.join(DATASET_DIR, "Voice")
MOTION_DIR = os.path.join(DATASET_DIR, "motion")
CSV_PATH = os.path.join(DATASET_DIR, "autism_dataset_index.csv")
FEATURES_DIR = os.path.join(DATASET_DIR, "features")

AUDIO_FEATURES_PATH = os.path.join(FEATURES_DIR, "audio_features.npy")
MOTION_FEATURES_PATH = os.path.join(FEATURES_DIR, "motion_features.npy")
LABELS_PATH = os.path.join(FEATURES_DIR, "labels.npy")

os.makedirs(FEATURES_DIR, exist_ok=True)

# ================= AUDIO FEATURE EXTRACTION =================
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"⚠️ Failed to process audio {file_path}: {e}")
        return None

# ================= MOTION FEATURE EXTRACTION =================
def extract_motion_features(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        all_parts = []

        # Iterate through all body parts in JSON (head, left_hand, right_hand, etc.)
        for part, frames in data.items():
            frames_array = np.array(frames)  # shape: (num_frames, 3)
            if frames_array.ndim == 2:
                mean_part = np.mean(frames_array, axis=0)
                all_parts.append(mean_part)
            elif frames_array.ndim == 1:
                all_parts.append(frames_array)

        if not all_parts:
            return None

        features = np.concatenate(all_parts, axis=0)
        return features.astype(np.float32)

    except Exception as e:
        print(f"⚠️ Failed to process motion {json_path}: {e}")
        return None

# ================= LOAD LABELS =================
df = pd.read_csv(CSV_PATH)
labels_dict = dict(zip(df['sample_id'], df['label']))

# ================= PROCESS DATA =================
audio_features, motion_features, labels = [], [], []

print("🎧 Extracting audio and motion features...")

for sample_id in df['sample_id']:
    # Audio
    audio_file = os.path.join(VOICE_DIR, f"{sample_id}.wav")
    if os.path.exists(audio_file):
        feats = extract_audio_features(audio_file)
        if feats is not None:
            audio_features.append(feats)
        else:
            audio_features.append(np.zeros(20, dtype=np.float32))
    else:
        audio_features.append(np.zeros(20, dtype=np.float32))

    # Motion
    motion_file = os.path.join(MOTION_DIR, f"{sample_id}.json")
    if os.path.exists(motion_file):
        feats = extract_motion_features(motion_file)
        if feats is not None:
            motion_features.append(feats)
        else:
            motion_features.append(np.zeros(60, dtype=np.float32))  # fallback dim
    else:
        motion_features.append(np.zeros(60, dtype=np.float32))  # fallback dim

    # Label
    labels.append(labels_dict[sample_id])

# Convert to numpy arrays
audio_features = np.array(audio_features, dtype=np.float32)
motion_features = np.array(motion_features, dtype=np.float32)
labels = np.array(labels)

# Save
np.save(AUDIO_FEATURES_PATH, audio_features)
np.save(MOTION_FEATURES_PATH, motion_features)
np.save(LABELS_PATH, labels)

print(f"✅ Audio features saved to: {AUDIO_FEATURES_PATH}")
print(f"✅ Motion features saved to: {MOTION_FEATURES_PATH}")
print(f"✅ Labels saved to: {LABELS_PATH}")
print("🎯 Dataset preparation completed successfully!")
