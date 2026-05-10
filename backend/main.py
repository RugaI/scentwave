# -*- coding: utf-8 -*-
"""
ScentWave FastAPI Backend
Run: uvicorn backend.main:app --reload --port 8000
"""
import sys, os, csv, json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="ScentWave API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from src.models.scentwave import ScentWave
from src.data.features import PRESET_SONGS

_MODEL: Optional[ScentWave] = None
_SONGS_DB: dict = {}
_SONGS_BY_ID: dict = {}

# Genre → estimated audio features (used when song isn't in our CSV)
_GENRE_FEATURES: dict = {
    "hip hop":    dict(danceability=0.78, energy=0.68, valence=0.55, speechiness=0.28, acousticness=0.12, instrumentalness=0.01, liveness=0.15),
    "rap":        dict(danceability=0.75, energy=0.70, valence=0.50, speechiness=0.32, acousticness=0.10, instrumentalness=0.01, liveness=0.14),
    "pop":        dict(danceability=0.68, energy=0.65, valence=0.62, speechiness=0.06, acousticness=0.22, instrumentalness=0.03, liveness=0.13),
    "rock":       dict(danceability=0.52, energy=0.76, valence=0.47, speechiness=0.05, acousticness=0.18, instrumentalness=0.10, liveness=0.19),
    "metal":      dict(danceability=0.42, energy=0.92, valence=0.35, speechiness=0.06, acousticness=0.05, instrumentalness=0.20, liveness=0.22),
    "jazz":       dict(danceability=0.55, energy=0.44, valence=0.60, speechiness=0.04, acousticness=0.68, instrumentalness=0.40, liveness=0.24),
    "classical":  dict(danceability=0.28, energy=0.22, valence=0.40, speechiness=0.04, acousticness=0.88, instrumentalness=0.82, liveness=0.10),
    "electronic": dict(danceability=0.74, energy=0.82, valence=0.55, speechiness=0.05, acousticness=0.07, instrumentalness=0.48, liveness=0.12),
    "r&b":        dict(danceability=0.72, energy=0.60, valence=0.58, speechiness=0.08, acousticness=0.25, instrumentalness=0.04, liveness=0.14),
    "soul":       dict(danceability=0.65, energy=0.55, valence=0.63, speechiness=0.05, acousticness=0.38, instrumentalness=0.05, liveness=0.18),
    "country":    dict(danceability=0.58, energy=0.65, valence=0.68, speechiness=0.04, acousticness=0.45, instrumentalness=0.03, liveness=0.13),
    "reggae":     dict(danceability=0.78, energy=0.58, valence=0.75, speechiness=0.08, acousticness=0.30, instrumentalness=0.05, liveness=0.17),
    "blues":      dict(danceability=0.52, energy=0.50, valence=0.48, speechiness=0.05, acousticness=0.55, instrumentalness=0.12, liveness=0.20),
    "funk":       dict(danceability=0.82, energy=0.72, valence=0.72, speechiness=0.08, acousticness=0.20, instrumentalness=0.15, liveness=0.18),
    "latin":      dict(danceability=0.80, energy=0.72, valence=0.72, speechiness=0.07, acousticness=0.20, instrumentalness=0.06, liveness=0.16),
    "indie":      dict(danceability=0.55, energy=0.60, valence=0.52, speechiness=0.05, acousticness=0.35, instrumentalness=0.12, liveness=0.16),
    "folk":       dict(danceability=0.48, energy=0.40, valence=0.55, speechiness=0.04, acousticness=0.72, instrumentalness=0.08, liveness=0.14),
    "ambient":    dict(danceability=0.30, energy=0.20, valence=0.42, speechiness=0.03, acousticness=0.75, instrumentalness=0.80, liveness=0.08),
}
_DEFAULT_FEATURES = dict(danceability=0.55, energy=0.60, valence=0.50, speechiness=0.08, acousticness=0.30, instrumentalness=0.10, liveness=0.15)

FEATURE_COLS = [
    "danceability","energy","key_norm","loudness_norm","mode",
    "speechiness","acousticness","instrumentalness","liveness",
    "valence","tempo_norm","time_sig_norm"
]


def get_model():
    global _MODEL
    if _MODEL is None:
        ckpt = Path(__file__).parent.parent / "checkpoints" / "best.pt"
        _MODEL = ScentWave.load(str(ckpt))
        _MODEL.eval()
        print("Model loaded.")
    return _MODEL


def get_songs_db():
    global _SONGS_DB, _SONGS_BY_ID
    if _SONGS_DB:
        return _SONGS_DB
    csv_path = Path(__file__).parent.parent / "data" / "songs.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            name   = row.get("name", "").strip().lower()
            artist = row.get("artist", "").strip().lower()
            _SONGS_DB[f"{name}||{artist}"] = row
            _SONGS_DB[name] = row
            tid = row.get("track_id", "")
            if tid:
                _SONGS_BY_ID[tid] = row
    print(f"Songs DB: {len(_SONGS_BY_ID)} tracks loaded.")
    return _SONGS_DB


def _get_spotify_client():
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        from dotenv import load_dotenv
        load_dotenv()
        cid = os.getenv("SPOTIFY_CLIENT_ID", "")
        sec = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        if not cid or not sec or cid == "your_client_id_here":
            return None
        return spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=cid, client_secret=sec))
    except Exception:
        return None


def _spotify_search(query: str, limit: int = 8) -> list:
    sp = _get_spotify_client()
    if not sp:
        return []
    try:
        res = sp.search(q=query, type="track", limit=limit, market="US")
        items = res.get("tracks", {}).get("items", [])
        results = []
        for t in items:
            results.append({
                "track_id": t["id"],
                "name": t["name"],
                "artist": t["artists"][0]["name"] if t["artists"] else "",
                "genre": "",
                "from_spotify": True,
                "artist_id": t["artists"][0]["id"] if t["artists"] else "",
                "popularity": t.get("popularity", 50),
                "mode": 1,
            })
        return results
    except Exception:
        return []


def _estimate_features_from_spotify(track_id: str, artist_id: str, popularity: int = 50, mode: int = 1) -> Optional[list]:
    """Estimate audio features for a Spotify track using genre + audio-analysis."""
    sp = _get_spotify_client()
    if not sp:
        return None

    genres = []
    tempo, key, loudness = 120.0, 5, -8.0

    try:
        artist_info = sp.artist(artist_id)
        genres = [g.lower() for g in artist_info.get("genres", [])]
    except Exception:
        pass

    try:
        analysis = sp.audio_analysis(track_id)
        t = analysis.get("track", {})
        tempo     = t.get("tempo", 120.0)
        key       = t.get("key", 5)
        mode      = t.get("mode", mode)
        loudness  = t.get("loudness", -8.0)
    except Exception:
        pass

    # Pick best matching genre preset
    feats = dict(_DEFAULT_FEATURES)
    for genre_key, genre_feats in _GENRE_FEATURES.items():
        if any(genre_key in g for g in genres):
            feats = dict(genre_feats)
            break

    # Adjust valence slightly by mode (major = happier)
    if mode == 0:
        feats["valence"] = max(0.05, feats["valence"] - 0.12)

    # Adjust energy by popularity (popular tracks tend toward higher energy)
    pop_boost = (popularity - 50) / 500.0  # ±0.10 max
    feats["energy"] = min(1.0, max(0.0, feats["energy"] + pop_boost))

    key_norm      = key / 11.0
    loudness_norm = min(1.0, max(0.0, (loudness + 60) / 60.0))
    tempo_norm    = min(1.0, tempo / 220.0)

    return [
        feats["danceability"],
        feats["energy"],
        key_norm,
        loudness_norm,
        float(mode),
        feats["speechiness"],
        feats["acousticness"],
        feats["instrumentalness"],
        feats["liveness"],
        feats["valence"],
        tempo_norm,
        0.9,  # time_sig_norm (4/4 default)
    ]


def _features_from_row(row: dict) -> Optional[list]:
    try:
        return [float(row[c]) for c in FEATURE_COLS]
    except (KeyError, ValueError):
        return None


class AnalyzeRequest(BaseModel):
    song: str
    artist: str = ""
    track_id: Optional[str] = None    # pass through from search result
    artist_id: Optional[str] = None   # pass through for feature estimation
    popularity: int = 50
    mode: int = 1


@app.on_event("startup")
async def startup():
    get_model()
    get_songs_db()


@app.get("/health")
def health():
    return {"status": "ok", "model": _MODEL is not None, "songs": len(_SONGS_BY_ID)}


@app.get("/search")
def search(q: str):
    db = get_songs_db()
    q_lower = q.lower().strip()
    if not q_lower:
        return []

    seen, results = set(), []

    # 1. Search local CSV
    for key, row in db.items():
        name   = row.get("name", "").lower()
        artist = row.get("artist", "").lower()
        if q_lower in name or q_lower in artist:
            tid = row.get("track_id", "")
            if tid and tid not in seen:
                seen.add(tid)
                results.append({
                    "track_id": tid,
                    "name":     row.get("name", ""),
                    "artist":   row.get("artist", ""),
                    "genre":    row.get("genre_hint", ""),
                })
        if len(results) >= 8:
            break

    # 2. Supplement with Spotify API if we have fewer than 5 local results
    if len(results) < 5:
        spotify_hits = _spotify_search(q, limit=8)
        for hit in spotify_hits:
            if hit["track_id"] not in seen:
                seen.add(hit["track_id"])
                results.append(hit)
            if len(results) >= 8:
                break

    # 3. Also check presets
    for preset_key, preset_data in PRESET_SONGS.items():
        pname = preset_data.get("name", preset_key).lower()
        if q_lower in pname or q_lower in preset_key:
            results.append({
                "track_id": None,
                "name":     preset_data.get("name", preset_key.title()),
                "artist":   preset_data.get("artist", ""),
                "genre":    "",
            })
        if len(results) >= 8:
            break

    return results[:8]


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    model  = get_model()
    db     = get_songs_db()

    song_lower   = req.song.strip().lower()
    artist_lower = req.artist.strip().lower()
    track_id     = req.track_id
    feats        = None
    display_name = req.song
    display_artist = req.artist

    # 1. Exact name+artist match in songs.csv
    key = f"{song_lower}||{artist_lower}"
    row = db.get(key) or db.get(song_lower)
    if row:
        feats          = _features_from_row(row)
        track_id       = track_id or row.get("track_id")
        display_name   = row.get("name", req.song)
        display_artist = row.get("artist", req.artist)

    # 2. Preset songs
    if feats is None:
        preset = PRESET_SONGS.get(song_lower)
        if not preset:
            for k, v in PRESET_SONGS.items():
                if song_lower in k or k in song_lower:
                    preset = v
                    break
        if preset:
            raw = preset.get("features", [])
            if raw and len(raw) == 12:
                feats = [float(x) for x in raw]
            display_name   = preset.get("name", req.song)
            display_artist = preset.get("artist", req.artist)

    # 3. Broader name search in CSV
    if feats is None:
        for _, row2 in db.items():
            if song_lower in row2.get("name", "").lower():
                feats          = _features_from_row(row2)
                track_id       = track_id or row2.get("track_id")
                display_name   = row2.get("name", req.song)
                display_artist = row2.get("artist", req.artist)
                break

    # 4. Spotify search + feature estimation for any song not in our CSV
    if feats is None:
        # If we have track_id + artist_id from the search result, use them directly
        artist_id = req.artist_id
        popularity = req.popularity
        mode = req.mode

        if not track_id or not artist_id:
            # Need to search Spotify first
            sp_results = _spotify_search(f"{req.song} {req.artist}", limit=3)
            if sp_results:
                best         = sp_results[0]
                track_id     = track_id or best["track_id"]
                artist_id    = artist_id or best.get("artist_id", "")
                popularity   = best.get("popularity", 50)
                mode         = best.get("mode", 1)
                display_name   = best["name"]
                display_artist = best["artist"]

        if track_id and artist_id:
            feats = _estimate_features_from_spotify(track_id, artist_id, popularity, mode)

    if feats is None:
        raise HTTPException(
            status_code=404,
            detail=f"Song '{req.song}' not found. Try typing a different spelling or including the artist name."
        )

    # 5. Run model
    result = model.predict(np.array(feats, dtype=np.float32))

    return {
        "song": {
            "name":       display_name,
            "artist":     display_artist,
            "track_id":   track_id,
            "spotify_url":  f"https://open.spotify.com/track/{track_id}" if track_id else None,
            "embed_url":    f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" if track_id else None,
        },
        "emotion":   result["emotion"],
        "retrieved": result["retrieved"],
        "generated": result["generated"],
    }
