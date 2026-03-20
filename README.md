# 🧠 Smart Autism Detection System

The Smart Autism Detection System is an AI-based web application designed to assist in the early identification of autism risk in children. It analyzes audio and video inputs to provide predictions and insights.

## 🚀 Features

- 🎧 Audio Analysis (WAV format)
- 🎥 Video Analysis (MP4 format)
- 🤖 AI-based Autism Risk Prediction
- 🔢 Optional Age Input for better accuracy
- 🎵 Built-in MP3 to WAV Converter
- 📊 Visual Result Representation (Charts)
- 💻 User-friendly Web Interface

## 🛠️ Technologies Used

- Python (Flask)
- TensorFlow / Machine Learning
- HTML, CSS, JavaScript
- FFmpeg (for audio conversion)

## ⚙️ How It Works

1. Upload child's audio (.wav) and video (.mp4)
2. Optionally enter age in months
3. Click on **Predict Risk**
4. System processes input using trained ML model
5. Displays autism risk score with visualization
6. If audio is in MP3, convert it using built-in tool

## 📂 Project Structure

- `app.py` – Main Flask application  
- `model/` – Trained ML model  
- `static/` – CSS, images, charts  
- `templates/` – HTML pages  
- `utils/` – Helper functions  

## ▶️ Run the Project

```bash
pip install -r requirements.txt
python app.py
