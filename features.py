import numpy as np
import librosa

def extract_features(audio_path, pause_start):
    """
    Extracts audio features from a 1.5-second window immediately preceding the pause.
    Strictly follows the causality rule: does not load any audio after pause_start.
    
    Args:
        audio_path (str): Path to the audio file.
        pause_start (float): Time in seconds where the pause begins.
        
    Returns:
        dict: Extracted features including RMS energy, Pitch (F0), and MFCCs.
    """
    # 1. Determine start and end times to strictly enforce causality
    start_time = max(0.0, pause_start - 1.5)
    end_time = pause_start
    duration = end_time - start_time
    
    # Handle edge case: non-positive duration
    if duration <= 0:
        return _get_empty_features()
        
    # 2. Load audio snippet
    try:
        # librosa.load with offset and duration guarantees we don't load future data
        y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=duration)
    except Exception as e:
        print(f"Warning: Failed to load {audio_path}: {e}")
        return _get_empty_features()
        
    # Handle edge case: empty audio returned
    if len(y) == 0:
        return _get_empty_features()
        
    features = {}
    
    # 3. RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    if len(rms) > 0:
        features['rms_mean'] = np.mean(rms)
        # Delta: simple slope from start to end of the sequence to capture fading volume
        features['rms_delta'] = rms[-1] - rms[0] if len(rms) > 1 else 0.0
    else:
        features['rms_mean'] = 0.0
        features['rms_delta'] = 0.0
        
    # 4. Pitch (F0) using pyin
    # fmin and fmax set to generic vocal ranges (C2 to C7 ~65Hz to ~2000Hz)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'), 
        sr=sr
    )
    
    # Filter out NaNs to compute statistics only on voiced frames
    if f0 is not None:
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            features['pitch_mean'] = np.mean(valid_f0)
            # Delta to capture falling intonation
            features['pitch_delta'] = valid_f0[-1] - valid_f0[0] if len(valid_f0) > 1 else 0.0
        else:
            features['pitch_mean'] = 0.0
            features['pitch_delta'] = 0.0
    else:
        features['pitch_mean'] = 0.0
        features['pitch_delta'] = 0.0
        
    # 5. MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfccs.shape[1] > 0:
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_var'] = np.var(mfccs[i])
    else:
        for i in range(13):
            features[f'mfcc_{i}_mean'] = 0.0
            features[f'mfcc_{i}_var'] = 0.0
            
    return features

def _get_empty_features():
    """Returns a zero-filled dictionary for edge cases."""
    features = {
        'rms_mean': 0.0,
        'rms_delta': 0.0,
        'pitch_mean': 0.0,
        'pitch_delta': 0.0,
    }
    for i in range(13):
        features[f'mfcc_{i}_mean'] = 0.0
        features[f'mfcc_{i}_var'] = 0.0
    return features

if __name__ == "__main__":
    # Simple self-test
    pass
