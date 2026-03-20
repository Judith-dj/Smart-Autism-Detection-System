import os

EXTRACTED_DIR = "dataset/autism-related-behavior-main"

for root, _, files in os.walk(EXTRACTED_DIR):
    for file in files:
        if file.lower().endswith((".wav", ".mp3", ".mp4", ".avi", ".mov")):
            print("Found media file:", os.path.join(root, file))
