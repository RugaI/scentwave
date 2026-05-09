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

# ── Load model once at startup ───────────────────────────────────────────────
from src.models.scentwave import ScentWave
from src.data.features import PRESET_SONGS

_MODEL: Optional[ScentWave] = None
_SONGS_DB: dict = {}   # name_artist → row
_SONGS_BY_ID: dict = {}  # track_id → row

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
            name = row.get("name", "").strip().lower()
            artist = row.get("artist", "").strip().lower()
            _SONGS_DB[f"{name}||{artist}"] = row
            _SONGS_DB[name] = row  # name-only fallback
            tid = row.get("track_id", "")
            if tid:
                _SONGS_BY_ID[tid] = row
    print(f"Songs DB: {len(_SONGS_BY_ID)} tracks loaded.")
    return _SONGS_DB

FEATURE_COLS = [
    "danceability","energy","key_norm","loudness_norm","mode",
    "speechiness","acousticness","instrumentalness","liveness",
    "valence","tempo_norm","time_sig_norm"
]

def _features_from_row(row: dict) -> Optional[list]:
    try:
        return [float(row[c]) for c in FEATURE_COLS]
    except (KeyError, ValueError):
        return None

def _spotify_search_track_id(song: str, artist: str = "") -> Optional[str]:
    """Get Spotify track_id via client credentials (search still works)."""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        from dotenv import load_dotenv
        load_dotenv()
        cid = os.getenv("SPOTIFY_CLIENT_ID", "")
        sec = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        if not cid or not sec or cid == "your_client_id_here":
            return None
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=cid, client_secret=sec))
        q = f"track:{song}" + (f" artist:{artist}" if artist else "")
        res = sp.search(q=q, type="track", limit=1, market="US")
        items = res.get("tracks", {}).get("items", [])
        if items:
            return items[0]["id"]
    except Exception:
        pass
    return None


class AnalyzeRequest(BaseModel):
    song: str
    artist: str = ""

class SearchRequest(BaseModel):
    q: str


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
    seen, results = set(), []
    for key, row in db.items():
        name = row.get("name", "").lower()
        artist = row.get("artist", "").lower()
        if q_lower in name or q_lower in artist:
            tid = row.get("track_id", "")
            if tid and tid not in seen:
                seen.add(tid)
                results.append({
                    "track_id": tid,
                    "name": row.get("name", ""),
                    "artist": row.get("artist", ""),
                    "genre": row.get("genre_hint", ""),
                })
        if len(results) >= 8:
            break

    # Also check presets
    for preset_key, preset_data in PRESET_SONGS.items():
        pname = preset_data.get("name", preset_key).lower()
        if q_lower in pname or q_lower in preset_key:
            results.append({
                "track_id": None,
                "name": preset_data.get("name", preset_key.title()),
                "artist": preset_data.get("artist", ""),
                "genre": "",
            })
        if len(results) >= 8:
            break

    return results[:8]


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    model = get_model()
    db = get_songs_db()

    song_lower = req.song.strip().lower()
    artist_lower = req.artist.strip().lower()
    track_id = None
    feats = None
    display_name = req.song
    display_artist = req.artist

    # 1. Try exact name+artist match in songs.csv
    key = f"{song_lower}||{artist_lower}"
    row = db.get(key) or db.get(song_lower)
    if row:
        feats = _features_from_row(row)
        track_id = row.get("track_id")
        display_name = row.get("name", req.song)
        display_artist = row.get("artist", req.artist)

    # 2. Try presets
    if feats is None:
        preset = PRESET_SONGS.get(song_lower)
        if not preset:
            # fuzzy preset match
            for k, v in PRESET_SONGS.items():
                if song_lower in k or k in song_lower:
                    preset = v
                    break
        if preset:
            raw = preset.get("features", [])
            if raw and len(raw) == 12:
                feats = [float(x) for x in raw]
            display_name = preset.get("name", req.song)
            display_artist = preset.get("artist", req.artist)

    # 3. Search songs.csv by name only (broader)
    if feats is None:
        for key2, row2 in db.items():
            if song_lower in row2.get("name", "").lower():
                feats = _features_from_row(row2)
                track_id = row2.get("track_id")
                display_name = row2.get("name", req.song)
                display_artist = row2.get("artist", req.artist)
                break

    if feats is None:
        raise HTTPException(status_code=404, detail=f"Song '{req.song}' not found. Try a different spelling or search.")

    # 4. Get Spotify track_id if we don't have one
    if not track_id:
        track_id = _spotify_search_track_id(display_name, display_artist)

    # 5. Run model
    result = model.predict(np.array(feats, dtype=np.float32))

    return {
        "song": {
            "name": display_name,
            "artist": display_artist,
            "track_id": track_id,
            "spotify_url": f"https://open.spotify.com/track/{track_id}" if track_id else None,
            "embed_url": f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" if track_id else None,
        },
        "emotion": result["emotion"],
        "retrieved": result["retrieved"],
        "generated": result["generated"],
    }
