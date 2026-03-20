# feature_utils.py  (safe fallback for Python 3.13)
import numpy as np
import librosa
import soundfile as sf
import cv2

# Try to import mediapipe; fall back if unavailable
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    HAS_MEDIAPIPE = True
except ImportError:
    print("⚠️  Mediapipe not found — using dummy video feature extractor.")
    HAS_MEDIAPIPE = False


# --- Audio: extract MFCCs ---
def extract_mfcc(path, sr=16000, n_mfcc=13, duration=5.0):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    target_len = int(sr * duration)
    if len(x) < target_len:
        x = np.pad(x, (0, target_len - len(x)))
    else:
        x = x[:target_len]
    mfcc = librosa.feature.mfcc(y=x.astype(float), sr=sr, n_mfcc=n_mfcc)
    return mfcc.T


# --- Video: use MediaPipe if available, otherwise zeros ---
def extract_video_keypoints(video_path, max_frames=150, downscale=0.5):
    if not HAS_MEDIAPIPE:
        # return dummy zeros if mediapipe unavailable
        return np.zeros((max_frames, 18))

    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
    pose = mp_pose.Pose(static_image_mode=False)
    features, frame_count = [], 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if downscale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * downscale), int(h * downscale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fm = face_mesh.process(rgb)
        ps = pose.process(rgb)
        feat = []
        if fm.multi_face_landmarks:
            lm = fm.multi_face_landmarks[0].landmark
            nose = lm[1]
            left_eye_idx = [33, 133, 160, 159]
            right_eye_idx = [263, 362, 387, 386]
            def avg_pts(idxs):
                xs = [lm[i].x for i in idxs]
                ys = [lm[i].y for i in idxs]
                return (np.mean(xs), np.mean(ys))
            le = avg_pts(left_eye_idx)
            re = avg_pts(right_eye_idx)
            feat += [nose.x, nose.y, le[0], le[1], re[0], re[1]]
        else:
            feat += [0] * 6
        if ps.pose_landmarks:
            pl = ps.pose_landmarks.landmark
            for idx in [11, 12, 23, 24]:
                p = pl[idx]
                feat += [p.x, p.y, p.z]
        else:
            feat += [0] * 12
        features.append(feat)
        frame_count += 1
    cap.release()
    face_mesh.close()
    pose.close()
    return np.array(features) if len(features) else np.zeros((1, 18))


# --- Behavioral features ---
def compute_behavioral_features(mfcc, keypoints):
    speech_energy = np.mean(np.abs(mfcc))
    pitch_proxy = np.mean(np.abs(mfcc[:, 0])) if mfcc.shape[1] > 0 else 0
    gaze_var = np.var(keypoints[:, 2:6]) if keypoints.shape[0] > 0 else 0
    return np.array([speech_energy, pitch_proxy, gaze_var], dtype=float)
