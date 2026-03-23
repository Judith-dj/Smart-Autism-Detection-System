import os
import pandas as pd
import subprocess
import zipfile
from pathlib import Path

DATASET_DIR = 'dataset/large_dataset'
os.makedirs(DATASET_DIR, exist_ok=True)

def download_kaggle_dataset():
    print('Setup Kaggle auth first: Download kaggle.json from kaggle.com/account, place in ~/.kaggle/')
    # Example datasets - replace with actual ~200 video ones
    datasets = [
        # Autism tabular - no video
        'vipoooool/autism-child-image-dataset',  # images
        # Gesture datasets as proxy
        'skylake101/chalearn-lap-isolated-gesture-video-dataset',  # videos
        # Add video-heavy
    ]
    for ds in datasets:
        subprocess.run(['kaggle', 'datasets', 'download', '-d', ds, '-p', DATASET_DIR])

def download_github_datasets():
    repos = [
        'https://github.com/pmorerio/video-gesture-autism.git',  # code only
        'https://github.com/ShijianDeng/AV-ASD.git',  # sample
    ]
    for repo in repos:
        subprocess.run(['git', 'clone', repo, os.path.join(DATASET_DIR, Path(repo).name)])

def synthetic_large_dataset(n=200):
    # Generate 200 realistic samples based on current
    import create_dummy_samples
    for i in range(101, n+1):
        label = '1' if i%4 !=0 else '0'  # ASD bias
        create_dummy_samples.create_sample(DATASET_DIR, f'child_{i:03d}', label)
    print(f'Generated {n} synthetic samples')

if __name__ == '__main__':
    print('1. Kaggle: Auth required')
    print('2. GitHub')
    print('3. Synthetic 200')
    synthetic_large_dataset(200)
    print('Run python sequential_prepare.py on new data, then train_sequential_model.py')

