import numpy as np
import librosa

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp
        
    return y, sr