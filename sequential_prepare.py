import os
import json
import numpy as np
import pandas as pd
from feature_utils import extract_mfcc, compute_behavioral_features

# Paths
DATASET_DIR = "dataset"
VOICE_DIR = os.path.join(DATASET_DIR, "Voice")
MOTION_DIR = os.path.join(DATASET_DIR, "motion")
CSV_PATH = os.path.join(DATASET_DIR, "autism_dataset_index.csv")
if not os.path.exists(CSV_PATH):
    LARGE_DIR = os.path.join(DATASET_DIR, "large_dataset")
    csv_lines = []
    for f in os.listdir(LARGE_DIR):
        if f.endswith('video.mp4'):
            sid = f.replace('child_', '').replace('_video.mp4', '')
            label_file = os.path.join(LARGE_DIR, f.replace('video.mp4', 'label.txt'))
            with open(label_file, 'r') as lf:
                label = lf.read().strip()
            csv_lines.append({'sample_id': sid, 'label': label, 'video_path': f'large_dataset/{f}', 'voice_path': f'large_dataset/{f.replace("video", "audio")}'})
    pd.DataFrame(csv_lines).to_csv(CSV_PATH, index=False)
    print(f'Generated CSV for {len(csv_lines)} large samples')
FEATURES_DIR = os.path.join(DATASET_DIR, "features")

os.makedirs(FEATURES_DIR, exist_ok=True)

SEQ_AUDIO_PATH = os.path.join(FEATURES_DIR, "sequential_audio.npy")
SEQ_MOTION_PATH = os.path.join(FEATURES_DIR, "sequential_motion.npy")
SEQ_BEH_PATH = os.path.join(FEATURES_DIR, "sequential_beh.npy")
LABELS_PATH = os.path.join(FEATURES_DIR, "labels.npy")

def pad_trunc(a, target_shape):
    """Pad or truncate array to target shape"""
    arr = np.zeros(target_shape, dtype=float)
    t = min(a.shape[0], target_shape[0])
    arr[:t] = a[:t]
    return arr

def parse_motion_json(json_path, max_frames=150):
    """Parse motion JSON to (max_frames, 18) keypoints"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    parts = ['head', 'left_hand', 'right_hand']
    frames = []
    num_frames = min(len(data['head']), max_frames)
    
    for f in range(num_frames):
        frame_kpts = []
        for part in parts:
            frame_pts = data[part][f][:6]  # Take first 6 points (x,y,z x3)
            frame_kpts.extend(frame_pts)
        # Pad to 18 if short
        while len(frame_kpts) < 18:
            frame_kpts.append(0.0)
        frame_kpts = frame_kpts[:18]
        frames.append(frame_kpts)
    
    return pad_trunc(np.array(frames), (max_frames, 18))

# Load index
df = pd.read_csv(CSV_PATH)
print(f"Processing {len(df)} samples...")

audio_feats, motion_feats, beh_feats, labels = [], [], [], []

for idx, row in df.iterrows():
    sample_id = row['sample_id']
    label = row['label']
    
    # Binary label: ASD=1, typical=0
    label_num = 1 if 'asd' in label else 0
    labels.append(label_num)
    
    audio_file = os.path.join(DATASET_DIR, row['voice'])
    motion_file = os.path.join(DATASET_DIR, row['motion'])
    
    # Audio MFCC (160,13)
    if os.path.exists(audio_file):
        mfcc = extract_mfcc(audio_file)
        audio_feat = pad_trunc(mfcc, (160, 13))
    else:
        audio_feat = np.zeros((160, 13))
    audio_feats.append(audio_feat)
    
    # Motion keypoints (150,18)
    if os.path.exists(motion_file):
        motion_feat = parse_motion_json(motion_file)
    else:
        motion_feat = np.zeros((150, 18))
    motion_feats.append(motion_feat)
    
    # Beh features
    beh_feat = compute_behavioral_features(audio_feat, motion_feat)
    beh_feats.append(beh_feat)
    
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(df)}: {sample_id}")

# Stack
audio_feats = np.array(audio_feats, dtype=np.float32)  # (100,160,13)
motion_feats = np.array(motion_feats, dtype=np.float32)  # (100,150,18)
beh_feats = np.array(beh_feats, dtype=np.float32)  # (100,3)
labels = np.array(labels, dtype=np.int32)  # (100,)

# Save
np.save(SEQ_AUDIO_PATH, audio_feats)
np.save(SEQ_MOTION_PATH, motion_feats)
np.save(SEQ_BEH_PATH, beh_feats)
np.save(LABELS_PATH, labels)

print(f"✅ Sequential features saved:")
print(f"  Audio: {audio_feats.shape} -> {SEQ_AUDIO_PATH}")
print(f"  Motion: {motion_feats.shape} -> {SEQ_MOTION_PATH}")
print(f"  Beh: {beh_feats.shape} -> {SEQ_BEH_PATH}")
print(f"  Labels (binary ASD/typical): {labels.shape}, {np.bincount(labels)}")
print("Run: python sequential_prepare.py")

