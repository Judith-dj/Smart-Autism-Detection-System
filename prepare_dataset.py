# prepare_dataset.py
import os
import pandas as pd
import urllib.request
from zipfile import ZipFile

# Create a folder for dataset
os.makedirs("dataset", exist_ok=True)

# Example: Download sample dataset zip (replace with actual dataset URL)
url = "https://github.com/ShijianDeng/AV-ASD/raw/main/sample_data.zip"
zip_path = os.path.join("dataset", "sample_data.zip")

# Download dataset
print("Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete!")

# Extract dataset
print("Extracting...")
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("dataset")
print("Extraction complete!")
