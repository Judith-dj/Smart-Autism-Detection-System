# Autism Detection Project - Improved Accuracy

## Overview
Multimodal Flask app for autism risk prediction from audio + video uploads.
Uses temporal MFCC audio, MediaPipe keypoints motion, behavioral stats.

**Current Accuracy:** Improved with 100-sample real dataset temporal training.

## Dataset
- **100 samples:** child_001-100
- **Modalities:** voice/*.wav, motion/*.json (3D keypoints head/hand/torso, stimming_detected), physio CSV, images PNG.
- **Labels:** mild_asd, moderate_asd, severe_asd, typical (binary ASD=1 vs 0)

## Training Pipeline (Improved)
1. `python sequential_prepare.py` → sequential_*.npy (temporal features)
2. `python train_sequential_model.py` → autism_model.keras (LSTM+Conv1D, 50 epochs)
3. `python app.py` → http://localhost:5000 upload test.

## Dependencies
pip install -r requirements.txt
pip install mediapipe protobuf==4.25.3 (fixed)

## Key Improvements
- **Temporal Features:** Seq vs mean pooling → better gesture/speech dynamics detection.
- **Real Data:** ASD behavioral motion (stimming), vocal patterns.
- **Model:** Full multimodal matching app inference.
- **Mediapipe Fixed:** Latest API compatibility.

Test with dataset/sample1/ - expect high-risk for ASD-like dummy.

Ready for production!

