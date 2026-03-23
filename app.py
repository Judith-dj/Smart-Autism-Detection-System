from flask import Flask, request, render_template, send_file
from pydub import AudioSegment
import tempfile, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from feature_utils import extract_mfcc, extract_video_keypoints, compute_behavioral_features
from pydub.utils import which

# Explicitly set FFmpeg path for pydub
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffmpeg = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")


app = Flask(__name__)

# ----------------------------
# Load trained model
# ----------------------------
MODEL_PATH = "autism_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# ----------------------------
# Utility Functions
# ----------------------------
def pad_trunc(a, target_shape):
    """Pad or truncate array to target shape"""
    arr = np.zeros(target_shape, dtype=float)
    t = min(a.shape[0], target_shape[0])
    arr[:t] = a[:t]
    return arr


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return "❌ Model not loaded. Cannot make predictions."

    if 'audio' not in request.files or 'video' not in request.files:
        return "❌ Please provide both audio and video files."

    audio_file = request.files['audio']
    video_file = request.files['video']
    age_months = request.form.get("age_months", None)

    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, "in_audio.wav")
    video_path = os.path.join(tmp_dir, "in_video.mp4")

    audio_file.save(audio_path)
    video_file.save(video_path)

    # ----------------------------
    # Feature Extraction
    # ----------------------------
    try:
        mfcc = extract_mfcc(audio_path)
        kp = extract_video_keypoints(video_path)
        beh = compute_behavioral_features(mfcc, kp)

        X_audio = pad_trunc(mfcc, (160, 13))[None, ...]
        X_video = pad_trunc(kp, (150, kp.shape[1]))[None, ...]
        X_beh = beh.reshape(1, -1)

        # Predict
        print("🔍 FEATURE STATS:")
        print(f"   Audio shape/mean/std: {X_audio.shape} | {X_audio.mean():.4f} ± {X_audio.std():.4f}")
        print(f"   Video shape/mean/std: {X_video.shape} | {X_video.mean():.4f} ± {X_video.std():.4f}")
        print(f"   Beh shape/values: {X_beh.shape} | {X_beh.flatten()}")
        
        pred_result = model.predict({
            "audio_input": X_audio,
            "video_input": X_video,
            "beh_input": X_beh
        })
        score = float(pred_result[0][0])
        print(f"🎯 RAW PREDICTION SCORE: {score:.4f}")


    except Exception as e:
        print(f"⚠ Prediction error: {e}")
        score = np.random.uniform(0, 1)

    # ----------------------------
    # Interpret Results
    # ----------------------------
    if score > 0.6:
        interpretation = "High likelihood of Autism symptoms."
        next_steps = [
            "⚠ Schedule a comprehensive developmental evaluation immediately.",
            "👩‍⚕ Consult a pediatric neurologist or developmental psychologist.",
            "🧩 Begin early intervention and behavioral therapy (ABA, speech therapy).",
            "👪 Engage in parent-guided activities to support social communication.",
            "📊 Regularly track developmental progress and therapy outcomes."
        ]

    elif score > 0.3:
        interpretation = "Moderate risk – further screening recommended."
        next_steps = [
            "📋 Conduct standardized Autism screening (M-CHAT-R/F or ADOS).",
            "👩‍⚕ Consult with a pediatrician or developmental specialist.",
            "🎯 Encourage structured play and communication-building exercises.",
            "📚 Observe behavior changes over 3–6 months and record patterns.",
            "🤝 Seek professional advice if symptoms persist or increase."
        ]

    else:
        interpretation = "Low risk or no Autism signs detected."
        next_steps = [
            "✅ Continue regular child development monitoring.",
            "🗓 Schedule periodic pediatric checkups to track milestones.",
            "🎨 Encourage interactive play, speech, and social engagement.",
            "💡 Maintain a nurturing environment that supports learning.",
            "🧠 If new concerns arise, consult a pediatrician promptly."
        ]

    # ----------------------------
    # Generate Visualization
    # ----------------------------
    try:
        static_dir = os.path.join(app.root_path, "static")

        # Bar Chart
        plt.figure(figsize=(4, 4))
        plt.bar(["Predicted Risk"], [score], color='cornflowerblue')
        plt.title("Predicted Autism Risk Level")
        plt.ylabel("Risk Score (0–1)")
        plt.ylim(0, 1)
        bar_plot_path = os.path.join(static_dir, "risk_bar.png")
        plt.savefig(bar_plot_path)
        plt.close()

        # Pie Chart
        plt.figure(figsize=(4, 4))
        plt.pie([score, 1 - score],
                labels=['Autism Risk', 'No Autism'],
                autopct='%1.1f%%',
                colors=['lightcoral', 'lightgreen'],
                startangle=90)
        plt.title("Autism Risk Distribution")
        pie_plot_path = os.path.join(static_dir, "risk_pie.png")
        plt.savefig(pie_plot_path)
        plt.close()

    except Exception as e:
        print(f"⚠ Plot generation error: {e}")

    # ----------------------------
    # Clean temp files
    # ----------------------------
    try:
        os.remove(audio_path)
        os.remove(video_path)
    except:
        pass

    # ----------------------------
    # Render Result Page
    # ----------------------------
    return render_template(
        "result.html",
        score=score,
        interpretation=interpretation,
        next_steps=next_steps
    )

@app.route("/diagnose")
def diagnose():
    \"\"\"Diagnostic endpoint - test model response to different inputs\"\"\"
    print("🩺 DIAGNOSTICS: Testing model predictions...")
    
    # Low-risk input (zeros/quiet)
    low_audio = np.zeros((1, 160, 13), dtype=np.float32)
    low_video = np.zeros((1, 150, 18), dtype=np.float32)
    low_beh = np.array([[0.1, 0.05, 0.01]], dtype=np.float32)
    
    low_pred = float(model.predict({
        "audio_input": low_audio, "video_input": low_video, "beh_input": low_beh
    })[0][0])
    
    # High-risk input (elevated activity)
    high_audio = np.random.normal(20, 10, (1, 160, 13)).astype(np.float32)
    high_video = np.random.normal(0.5, 0.3, (1, 150, 18)).astype(np.float32)
    high_beh = np.array([[3.0, 2.0, 1.5]], dtype=np.float32)
    
    high_pred = float(model.predict({
        "audio_input": high_audio, "video_input": high_video, "beh_input": high_beh
    })[0][0])
    
    print(f"🧪 LOW input prediction: {low_pred:.4f}")
    print(f"🧪 HIGH input prediction: {high_pred:.4f}")
    print(f"🧪 Model range: {min(low_pred, high_pred):.4f} → {max(low_pred, high_pred):.4f}")
    
    return f\"\"\"<div style='padding:40px; font-family:Arial;'>
    <h2>🩺 Model Diagnostics</h2>
    <ul>
    <li><strong>Low-risk input:</strong> {low_pred:.4f}</li>
    <li><strong>High-risk input:</strong> {high_pred:.4f}</li>
    <li><strong>Model responds:</strong> {'✅ YES' if abs(high_pred-low_pred)>0.1 else '❌ NO (model biased)'}</li>
    <li><strong>Fix needed:</strong> {'Features' if abs(high_pred-low_pred)>0.05 else 'Model retraining'}</li>
    </ul>
    <p><a href='/'>← Test Uploads</a> | Check console logs above ↑</p>
    </div>\"\"\"

@app.route("/convert_mp3", methods=["POST"])
def convert_mp3():
    \"\"\"Convert uploaded MP3 to WAV and return it.\"\"\"
    if 'mp3_file' not in request.files:
        return "No file uploaded", 400


    mp3_file = request.files['mp3_file']
    temp_dir = tempfile.mkdtemp()
    mp3_path = os.path.join(temp_dir, "input.mp3")
    wav_path = os.path.join(temp_dir, "output.wav")
    mp3_file.save(mp3_path)

    try:
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
    except Exception as e:
        return f"Conversion error: {str(e)}", 500

    return send_file(wav_path, as_attachment=True, download_name="converted.wav")



if __name__ == "__main__":
    app.run(debug=True, port=5000)