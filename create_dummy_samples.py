# create_dummy_samples.py
import numpy as np
import soundfile as sf
import cv2
import os

def make_audio(path, duration=5, sr=16000):
    t = np.linspace(0, duration, int(sr*duration), False)
    tone = 0.05 * np.sin(2*np.pi*220*t)   # quiet sine
    sf.write(path, tone, sr)

def make_video(path, width=640, height=480, frames=150, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'m','p','4','v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, f'Frame {i+1}', (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        out.write(frame)
    out.release()

def create_sample(base_dir, name, label='0'):
    d = os.path.join(base_dir, name)
    os.makedirs(d, exist_ok=True)
    make_audio(os.path.join(d,'audio.wav'))
    make_video(os.path.join(d,'video.mp4'))
    with open(os.path.join(d,'label.txt'),'w') as f:
        f.write(label)

if __name__ == '__main__':
    base='dataset'
    create_sample(base,'sample1','0')
    create_sample(base,'sample2','1')
    print('Created 2 sample folders in dataset/')
