# model.py
import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_multimodal_model(mfcc_time_steps=160, mfcc_dim=13, video_frames=150, video_feat_dim=18, behav_dim=3):
    # Audio branch: 1D conv over time
    audio_input = Input(shape=(mfcc_time_steps, mfcc_dim), name='audio_input')
    x = layers.Conv1D(64, 5, activation='relu')(audio_input)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    audio_out = layers.Dense(64, activation='relu')(x)

    # Video branch: temporal LSTM over keypoints
    video_input = Input(shape=(video_frames, video_feat_dim), name='video_input')
    y = layers.Masking()(video_input)
    y = layers.LSTM(64, return_sequences=False)(y)
    video_out = layers.Dense(64, activation='relu')(y)

    # Behavioral features branch
    beh_input = Input(shape=(behav_dim,), name='beh_input')
    b = layers.Dense(32, activation='relu')(beh_input)
    b = layers.Dense(16, activation='relu')(b)

    # Concatenate
    concat = layers.concatenate([audio_out, video_out, b])
    z = layers.Dense(64, activation='relu')(concat)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    # single output risk score (research-only)
    output = layers.Dense(1, activation='sigmoid', name='risk_score')(z)

    model = models.Model(inputs=[audio_input, video_input, beh_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model
