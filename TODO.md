# Autism Risk Prediction Fix - TODO
Status: [IN PROGRESS]

## Approved Plan Steps:

### 1. [PENDING] Create diagnostic logging in app.py
- Add print statements for score, feature means/stds
- Add /diagnose route to test model with synthetic data
- ✅ Done by BLACKBOXAI

### 2. [PENDING] Improve feature_utils.py robustness
- Normalize MFCC features
- Add realistic dummy keypoints if no MediaPipe
- Enhance behavioral features

### 3. [PENDING] Ensure MediaPipe installation
- `pip install mediapipe`
- Test video feature extraction

### 4. [PENDING] Diagnose & fix model bias
- Check label balance in sequential_prepare.py output
- Test model.predict on known ASD/typical samples
- Retrain if imbalanced (run train_sequential_model.py)

### 5. [PENDING] Test end-to-end
- Run `python app.py`
- Upload sample1/audio.wav + video.mp4
- Verify varying risk outputs
- Check http://localhost:5000

### 6. [PENDING] Production fixes
- Lower thresholds if needed
- Add input validation
- Clean up diagnostics

**Next: Diagnostic edits complete → Test → Report results**

