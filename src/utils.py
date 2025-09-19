import os
import json
import joblib
import numpy as np
from datetime import datetime
from src.logger import logging
from src.exceptions import CustomException
import sys
import subprocess
import tempfile
import cv2
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf



# -------------------------------
# File + Directory Utils
# -------------------------------
def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj, path: str):
    """Save dict/list as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    logging.info(f"JSON saved → {path}")

def load_json(path: str):
    """Load JSON into dict/list."""
    with open(path, "r") as f:
        return json.load(f)

# -------------------------------
# Model Utils
# -------------------------------
def save_object(model, path: str):
    """Save sklearn/torch model."""
    try:
        ensure_dir(os.path.dirname(path))
        joblib.dump(model, path)
        logging.info(f"Model saved → {path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(path: str):
    """Load sklearn/torch model."""
    return joblib.load(path)

# -------------------------------
# Numpy / Features
# -------------------------------
def load_npz_features(path: str):
    """Load .npz feature file and return X, y."""
    d = np.load(path, allow_pickle=True)
    x = np.concatenate([d["x_audio"], d["x_video"], d["x_text"]]).astype(np.float32)
    y = d["y"].item() if d["y"].shape == () else str(d["y"])
    return x, y

# -------------------------------
# Logging Helpers
# -------------------------------
def timestamp():
    """Return current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_wav16k(video_path: str) -> str:
    """
    Convert video/audio file into temporary 16kHz mono WAV.
    """
    try:
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        cmd = ["ffmpeg", "-loglevel", "error", "-y",
               "-i", video_path, "-ar", "16000", "-ac", "1", tmp_wav]
        subprocess.run(cmd, check=True)
        return tmp_wav
    except Exception as e:
        raise CustomException(f"FFmpeg failed on {video_path}: {e}", sys)


def extract_audio_features(wav_path: str) -> np.ndarray:
    """
    Extract audio embedding using Wav2Vec2 (mean pooled).
    """
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        speech, sr = sf.read(wav_path)
        inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        return emb.astype(np.float32)
    except Exception as e:
        raise CustomException(f"Audio feature extraction failed: {e}", sys)


def extract_video_features(video_path: str, num_frames: int = 16) -> np.ndarray:
    """
    Simple baseline: sample frames and flatten grayscale pixels.
    Replace later with CLIP or OpenFace features.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(0, total - 1), num_frames).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64)).flatten()
            frames.append(resized)
        cap.release()
        if len(frames) == 0:
            return np.zeros(64 * 64, dtype=np.float32)
        return np.mean(frames, axis=0).astype(np.float32)
    except Exception as e:
        raise CustomException(f"Video feature extraction failed: {e}", sys)


def extract_features_from_file(file_path: str) -> np.ndarray:
    """
    Entry point for prediction: raw video/audio file → fused [audio+video+text] feature.
    """
    try:
        logging.info(f"Extracting features from {file_path}")

        # Audio
        wav_path = ensure_wav16k(file_path)
        a_feat = extract_audio_features(wav_path)

        # Video
        v_feat = extract_video_features(file_path)

        # Text placeholder (not available for uploads)
        t_feat = np.zeros(1, dtype=np.float32)

        fused = np.concatenate([a_feat, v_feat, t_feat]).astype(np.float32)

        os.remove(wav_path)
        return fused

    except Exception as e:
        raise CustomException(e, sys)

