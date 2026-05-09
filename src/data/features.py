# -*- coding: utf-8 -*-
"""
Audio feature extraction.
Two backends:
  - Spotify API  (12 normalized features)  — needs API key
  - librosa      (12 equivalent features)  — works on any audio file
Both produce the same 12-dim vector so the model is backend-agnostic.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

FEATURE_NAMES = [
    "danceability", "energy", "key_norm", "loudness_norm",
    "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo_norm", "time_sig_norm",
]

# ── Spotify backend ──────────────────────────────────────────────────────────

def _get_spotify_client():
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        from dotenv import load_dotenv
        load_dotenv()
        cid = os.getenv("SPOTIFY_CLIENT_ID")
        secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        if not cid or not secret:
            return None
        return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=cid, client_secret=secret))
    except Exception:
        return None


def features_from_spotify(song_name: str, artist: str = "") -> Optional[np.ndarray]:
    sp = _get_spotify_client()
    if sp is None:
        return None
    query = f"track:{song_name}"
    if artist:
        query += f" artist:{artist}"
    results = sp.search(q=query, type="track", limit=1)
    tracks = results.get("tracks", {}).get("items", [])
    if not tracks:
        return None
    track_id = tracks[0]["id"]
    af = sp.audio_features(track_id)[0]
    if af is None:
        return None
    vec = np.array([
        af["danceability"],
        af["energy"],
        af["key"] / 11.0,
        (af["loudness"] + 60.0) / 60.0,
        float(af["mode"]),
        af["speechiness"],
        af["acousticness"],
        af["instrumentalness"],
        af["liveness"],
        af["valence"],
        min(af["tempo"] / 220.0, 1.0),
        (af["time_signature"] - 1) / 6.0,
    ], dtype=np.float32)
    meta = {
        "name":    tracks[0]["name"],
        "artist":  tracks[0]["artists"][0]["name"],
        "album":   tracks[0]["album"]["name"],
        "preview": tracks[0].get("preview_url"),
        "image":   tracks[0]["album"]["images"][0]["url"] if tracks[0]["album"]["images"] else None,
    }
    return np.clip(vec, 0.0, 1.0), meta


# ── librosa backend ──────────────────────────────────────────────────────────

def features_from_audio(audio_path: str) -> Optional[np.ndarray]:
    try:
        import librosa
    except ImportError:
        raise ImportError("Install librosa: pip install librosa")

    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms       = float(np.mean(librosa.feature.rms(y=y)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    sc        = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    sr_       = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_m  = float(np.mean(chroma))
    chroma_std= float(np.std(chroma))
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_m    = np.mean(mfccs, axis=1)

    # Map to the same 12-dim vector schema
    danceability    = float(np.clip(1.0 - zcr * 50, 0, 1))
    energy          = float(np.clip(rms * 20, 0, 1))
    key_norm        = float(np.clip(chroma_m, 0, 1))
    loudness_norm   = float(np.clip(rms * 15, 0, 1))
    mode            = float(chroma_std > 0.2)
    speechiness     = float(np.clip(zcr * 10, 0, 1))
    acousticness    = float(np.clip(1.0 - energy, 0, 1))
    instrumentalness= float(np.clip(1.0 - speechiness, 0, 1))
    liveness        = float(np.clip(np.std(mfcc_m[:5]) / 50, 0, 1))
    valence         = float(np.clip((mfcc_m[1] + 300) / 600, 0, 1))
    tempo_norm      = float(np.clip(float(tempo) / 220.0, 0, 1))
    time_sig_norm   = 0.5

    vec = np.array([
        danceability, energy, key_norm, loudness_norm, mode,
        speechiness, acousticness, instrumentalness, liveness,
        valence, tempo_norm, time_sig_norm,
    ], dtype=np.float32)

    path = Path(audio_path)
    meta = {"name": path.stem, "artist": "Unknown", "album": "", "preview": None, "image": None}
    return np.clip(vec, 0.0, 1.0), meta


# ── Synthetic features (for testing / demo without API) ─────────────────────

PRESET_SONGS = {
    # ── CLASSICAL ────────────────────────────────────────────────────────────
    "moonlight sonata":         [0.12, 0.15, 0.45, 0.20, 0.0, 0.04, 0.95, 0.90, 0.10, 0.18, 0.30, 0.5],
    "clair de lune":            [0.20, 0.12, 0.40, 0.18, 0.0, 0.03, 0.98, 0.95, 0.08, 0.22, 0.25, 0.5],
    "four seasons spring":      [0.65, 0.72, 0.55, 0.65, 1.0, 0.04, 0.90, 0.88, 0.15, 0.80, 0.60, 0.5],
    "canon in d":               [0.35, 0.25, 0.42, 0.28, 1.0, 0.03, 0.97, 0.95, 0.10, 0.65, 0.38, 0.5],
    "symphony no 9 beethoven":  [0.30, 0.85, 0.58, 0.88, 1.0, 0.05, 0.60, 0.75, 0.25, 0.55, 0.75, 0.5],
    "nocturne op 9 chopin":     [0.15, 0.10, 0.38, 0.15, 0.0, 0.03, 0.99, 0.97, 0.06, 0.25, 0.22, 0.5],

    # ── JAZZ ─────────────────────────────────────────────────────────────────
    "take five":                [0.55, 0.40, 0.50, 0.45, 0.0, 0.04, 0.75, 0.70, 0.20, 0.55, 0.52, 0.5],
    "so what miles davis":      [0.40, 0.35, 0.45, 0.40, 0.0, 0.04, 0.80, 0.78, 0.18, 0.45, 0.45, 0.5],
    "fly me to the moon":       [0.62, 0.50, 0.50, 0.52, 1.0, 0.05, 0.70, 0.65, 0.20, 0.75, 0.55, 0.5],
    "autumn leaves":            [0.35, 0.28, 0.42, 0.30, 0.0, 0.03, 0.85, 0.80, 0.12, 0.30, 0.40, 0.5],
    "feeling good nina simone": [0.50, 0.55, 0.48, 0.58, 0.0, 0.06, 0.65, 0.50, 0.22, 0.70, 0.58, 0.5],

    # ── POP ──────────────────────────────────────────────────────────────────
    "blinding lights":          [0.68, 0.82, 0.55, 0.80, 1.0, 0.05, 0.15, 0.00, 0.35, 0.60, 0.72, 0.5],
    "shape of you":             [0.82, 0.65, 0.55, 0.70, 1.0, 0.08, 0.10, 0.00, 0.12, 0.93, 0.77, 0.5],
    "bad guy":                  [0.70, 0.43, 0.50, 0.55, 0.0, 0.10, 0.35, 0.00, 0.08, 0.56, 0.66, 0.5],
    "someone like you":         [0.40, 0.30, 0.42, 0.35, 1.0, 0.03, 0.80, 0.00, 0.10, 0.15, 0.42, 0.5],
    "rolling in the deep":      [0.60, 0.78, 0.52, 0.78, 1.0, 0.06, 0.20, 0.00, 0.18, 0.40, 0.68, 0.5],
    "uptown funk":              [0.90, 0.88, 0.58, 0.85, 1.0, 0.08, 0.05, 0.00, 0.10, 0.95, 0.85, 0.5],
    "perfect ed sheeran":       [0.48, 0.35, 0.48, 0.38, 1.0, 0.04, 0.72, 0.00, 0.12, 0.80, 0.48, 0.5],
    "levitating dua lipa":      [0.80, 0.80, 0.52, 0.78, 1.0, 0.06, 0.08, 0.00, 0.15, 0.82, 0.78, 0.5],
    "watermelon sugar":         [0.82, 0.70, 0.52, 0.72, 1.0, 0.06, 0.18, 0.00, 0.15, 0.88, 0.72, 0.5],
    "as it was harry styles":   [0.62, 0.68, 0.50, 0.68, 0.0, 0.05, 0.25, 0.00, 0.12, 0.65, 0.70, 0.5],

    # ── ROCK ─────────────────────────────────────────────────────────────────
    "bohemian rhapsody":        [0.45, 0.70, 0.60, 0.72, 1.0, 0.06, 0.40, 0.60, 0.20, 0.45, 0.65, 0.5],
    "hotel california":         [0.55, 0.65, 0.50, 0.68, 1.0, 0.05, 0.35, 0.55, 0.25, 0.50, 0.58, 0.5],
    "smells like teen spirit":  [0.50, 0.91, 0.60, 0.88, 1.0, 0.10, 0.01, 0.00, 0.30, 0.60, 0.72, 0.5],
    "stairway to heaven":       [0.38, 0.60, 0.52, 0.65, 1.0, 0.05, 0.55, 0.60, 0.20, 0.42, 0.55, 0.5],
    "wish you were here":       [0.42, 0.35, 0.48, 0.40, 1.0, 0.04, 0.70, 0.68, 0.15, 0.28, 0.45, 0.5],
    "sweet child o mine":       [0.58, 0.75, 0.55, 0.75, 1.0, 0.06, 0.15, 0.10, 0.22, 0.62, 0.70, 0.5],
    "november rain":            [0.35, 0.45, 0.50, 0.50, 1.0, 0.05, 0.55, 0.40, 0.18, 0.22, 0.48, 0.5],
    "paint it black":           [0.42, 0.75, 0.55, 0.72, 0.0, 0.06, 0.25, 0.30, 0.22, 0.18, 0.68, 0.5],

    # ── HIP-HOP / R&B ────────────────────────────────────────────────────────
    "gods plan drake":          [0.62, 0.58, 0.50, 0.62, 0.0, 0.18, 0.15, 0.00, 0.18, 0.65, 0.72, 0.5],
    "alright kendrick":         [0.75, 0.65, 0.52, 0.70, 0.0, 0.22, 0.10, 0.00, 0.20, 0.72, 0.78, 0.5],
    "bitch better have my money": [0.78, 0.82, 0.55, 0.82, 0.0, 0.15, 0.05, 0.00, 0.12, 0.55, 0.80, 0.5],
    "redbone childish gambino": [0.55, 0.45, 0.48, 0.50, 0.0, 0.08, 0.40, 0.00, 0.15, 0.62, 0.60, 0.5],
    "lose yourself eminem":     [0.62, 0.85, 0.58, 0.85, 0.0, 0.28, 0.05, 0.00, 0.25, 0.45, 0.82, 0.5],
    "no role modelz":           [0.58, 0.62, 0.50, 0.65, 0.0, 0.20, 0.12, 0.00, 0.18, 0.55, 0.72, 0.5],

    # ── ELECTRONIC / EDM ─────────────────────────────────────────────────────
    "one more time daft punk":  [0.80, 0.88, 0.55, 0.85, 1.0, 0.05, 0.08, 0.00, 0.20, 0.90, 0.85, 0.5],
    "strobe deadmau5":          [0.55, 0.65, 0.50, 0.68, 0.0, 0.04, 0.30, 0.90, 0.15, 0.55, 0.78, 0.5],
    "levels avicii":            [0.75, 0.85, 0.55, 0.82, 1.0, 0.04, 0.10, 0.00, 0.20, 0.80, 0.88, 0.5],
    "animals martin garrix":    [0.70, 0.92, 0.55, 0.90, 0.0, 0.05, 0.05, 0.00, 0.25, 0.62, 0.90, 0.5],
    "midnight city m83":        [0.62, 0.70, 0.52, 0.72, 0.0, 0.04, 0.35, 0.60, 0.18, 0.65, 0.75, 0.5],
    "sandstorm darude":         [0.72, 0.90, 0.55, 0.88, 0.0, 0.05, 0.05, 0.00, 0.28, 0.55, 0.88, 0.5],

    # ── AMBIENT / LOFI ────────────────────────────────────────────────────────
    "lofi chill beats":         [0.55, 0.25, 0.50, 0.30, 0.0, 0.04, 0.92, 0.80, 0.10, 0.40, 0.38, 0.5],
    "weightless marconi union": [0.20, 0.08, 0.35, 0.12, 0.0, 0.03, 0.99, 0.98, 0.05, 0.35, 0.18, 0.5],
    "experience ludovico":      [0.18, 0.12, 0.38, 0.15, 0.0, 0.03, 0.97, 0.95, 0.06, 0.30, 0.22, 0.5],
    "avril 14th aphex twin":    [0.22, 0.10, 0.40, 0.12, 0.0, 0.03, 0.98, 0.97, 0.05, 0.38, 0.20, 0.5],

    # ── SOUL / BLUES ─────────────────────────────────────────────────────────
    "superstition stevie wonder": [0.80, 0.82, 0.55, 0.80, 1.0, 0.08, 0.15, 0.00, 0.22, 0.85, 0.80, 0.5],
    "what a wonderful world":   [0.40, 0.22, 0.45, 0.28, 1.0, 0.04, 0.85, 0.00, 0.12, 0.88, 0.35, 0.5],
    "hallelujah jeff buckley":  [0.28, 0.22, 0.42, 0.25, 1.0, 0.04, 0.88, 0.00, 0.10, 0.18, 0.32, 0.5],
    "the thrill is gone bb king": [0.38, 0.48, 0.48, 0.52, 0.0, 0.05, 0.60, 0.40, 0.18, 0.25, 0.52, 0.5],

    # ── LATIN / WORLD ────────────────────────────────────────────────────────
    "la bamba":                 [0.88, 0.88, 0.58, 0.82, 1.0, 0.08, 0.20, 0.00, 0.28, 0.88, 0.88, 0.5],
    "despacito":                [0.82, 0.72, 0.55, 0.72, 1.0, 0.08, 0.18, 0.00, 0.20, 0.90, 0.78, 0.5],
    "samba de uma nota so":     [0.78, 0.68, 0.52, 0.68, 1.0, 0.06, 0.45, 0.40, 0.18, 0.82, 0.72, 0.5],
    "oye como va":              [0.80, 0.78, 0.55, 0.75, 0.0, 0.06, 0.30, 0.25, 0.22, 0.85, 0.80, 0.5],

    # ── METAL ────────────────────────────────────────────────────────────────
    "master of puppets":        [0.28, 0.95, 0.62, 0.92, 0.0, 0.08, 0.02, 0.00, 0.35, 0.12, 0.92, 0.5],
    "paranoid black sabbath":   [0.38, 0.90, 0.60, 0.88, 0.0, 0.08, 0.05, 0.00, 0.30, 0.18, 0.88, 0.5],
    "through the fire flames":  [0.30, 0.95, 0.62, 0.92, 0.0, 0.06, 0.02, 0.00, 0.38, 0.15, 0.95, 0.5],

    # ── COUNTRY ──────────────────────────────────────────────────────────────
    "jolene dolly parton":      [0.48, 0.38, 0.48, 0.42, 1.0, 0.05, 0.72, 0.00, 0.15, 0.25, 0.45, 0.5],
    "the gambler kenny rogers": [0.45, 0.35, 0.45, 0.40, 1.0, 0.05, 0.75, 0.00, 0.18, 0.55, 0.42, 0.5],
    "take me home country roads":[0.65, 0.55, 0.50, 0.58, 1.0, 0.06, 0.62, 0.00, 0.18, 0.78, 0.58, 0.5],
}

def features_from_preset(song_name: str) -> Optional[tuple]:
    key = song_name.lower().strip()
    for preset_key, vec in PRESET_SONGS.items():
        if preset_key in key or key in preset_key:
            arr = np.array(vec, dtype=np.float32)
            meta = {"name": song_name, "artist": "Demo", "album": "", "preview": None, "image": None}
            return arr, meta
    return None


def extract_features(song_name: str = None, artist: str = "",
                     audio_path: str = None) -> tuple:
    """
    Main entry. Returns (feature_vector [12], meta_dict).
    Priority: Spotify > librosa > preset > random.
    """
    if audio_path:
        result = features_from_audio(audio_path)
        if result is not None:
            return result

    if song_name:
        preset = features_from_preset(song_name)
        if preset is not None:
            return preset

        result = features_from_spotify(song_name, artist)
        if result is not None:
            return result

    # Fallback: generate plausible random features
    rng = np.random.default_rng(abs(hash(song_name or "random")) % (2**31))
    vec = rng.uniform(0.1, 0.9, size=12).astype(np.float32)
    meta = {"name": song_name or "Unknown", "artist": artist or "Unknown",
            "album": "", "preview": None, "image": None}
    return vec, meta
