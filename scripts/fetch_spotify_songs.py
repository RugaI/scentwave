# -*- coding: utf-8 -*-
"""
ScentWave — Spotify Song Feature Fetcher
Pulls audio features for thousands of songs and saves to data/songs.csv.
This becomes real training data (songs ↔ audio features).

Usage:
    python scripts/fetch_spotify_songs.py --count 5000
    python scripts/fetch_spotify_songs.py --playlist 37i9dQZF1DXcBWIGoYBM5M
    python scripts/fetch_spotify_songs.py --search "jazz classics" --limit 200

Requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env
"""

import sys, os, csv, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

OUT_CSV = Path(__file__).parent.parent / "data" / "songs.csv"

FIELDNAMES = [
    "track_id","name","artist","album","year","genre_hint",
    "danceability","energy","key_norm","loudness_norm","mode",
    "speechiness","acousticness","instrumentalness","liveness",
    "valence","tempo_norm","time_sig_norm",
    "popularity","duration_ms",
]

# Broad search queries to pull diverse songs
SEARCH_QUERIES = [
    # Genres
    "genre:classical", "genre:jazz", "genre:blues", "genre:country",
    "genre:rock", "genre:metal", "genre:punk", "genre:indie",
    "genre:pop", "genre:r-n-b", "genre:hip-hop", "genre:rap",
    "genre:electronic", "genre:ambient", "genre:techno", "genre:house",
    "genre:latin", "genre:reggae", "genre:soul", "genre:funk",
    # Decades
    "year:1960-1969", "year:1970-1979", "year:1980-1989",
    "year:1990-1999", "year:2000-2009", "year:2010-2019", "year:2020-2024",
    # Moods/vibes
    "mood sad", "mood happy", "mood dark", "mood romantic",
    "chill relaxing", "workout energy", "focus study", "late night",
    "rain melancholy", "summer vibes", "winter cozy",
    # Instruments
    "piano solo", "guitar acoustic", "violin classical", "jazz trumpet",
    "drum bass electronic", "synthesizer ambient",
]

# Curated public playlists (Spotify playlist IDs)
PLAYLIST_IDS = [
    "37i9dQZF1DXcBWIGoYBM5M",  # Today's Top Hits
    "37i9dQZF1DX0XUsuxWHRQd",  # RapCaviar
    "37i9dQZF1DX4sWSpwq3LiO",  # Peaceful Piano
    "37i9dQZF1DX4dyzvuaRJ0n",  # mint (fresh pop)
    "37i9dQZF1DWXRqgorJj26U",  # Rock Classics
    "37i9dQZF1DX5trt9i14X7j",  # Jazz Classics
    "37i9dQZF1DX8tZsk68tuDw",  # Dance Hits
    "37i9dQZF1DX1lVhptIYRda",  # Hot Country
    "37i9dQZF1DX10zKzsJ2jva",  # Viva Latino
    "37i9dQZF1DWZeKCadgRdKQ",  # Deep Focus
    "37i9dQZF1DX4WYpdgoIcn6",  # Chill Hits
    "37i9dQZF1DX76Wlfdnj7AP",  # Beast Mode
    "37i9dQZF1DX9sIqqvKsjEJ",  # Songs to Sing in the Car
    "37i9dQZF1DX3rxVfibe1L0",  # Mood Booster
    "37i9dQZF1DWT7XSlIpX3s6",  # Dark & Stormy
]


def get_sp():
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    cid    = os.getenv("SPOTIFY_CLIENT_ID")
    secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or cid == "your_client_id_here":
        raise ValueError(
            "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env\n"
            "Get free credentials at: https://developer.spotify.com/dashboard"
        )
    # OAuth flow — audio features require user auth since Spotify's Nov 2024 API change.
    # First run opens a browser; token is cached in .cache for future runs.
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=cid,
        client_secret=secret,
        redirect_uri="http://localhost:8888/callback",
        scope="user-read-private user-top-read",
        cache_path=str(Path(__file__).parent.parent / ".spotify_cache"),
        open_browser=True,
    ))


def normalize_features(af: dict) -> dict:
    return {
        "danceability":      af["danceability"],
        "energy":            af["energy"],
        "key_norm":          af["key"] / 11.0,
        "loudness_norm":     min((af["loudness"] + 60.0) / 60.0, 1.0),
        "mode":              float(af["mode"]),
        "speechiness":       af["speechiness"],
        "acousticness":      af["acousticness"],
        "instrumentalness":  af["instrumentalness"],
        "liveness":          af["liveness"],
        "valence":           af["valence"],
        "tempo_norm":        min(af["tempo"] / 220.0, 1.0),
        "time_sig_norm":     (af["time_signature"] - 1) / 6.0,
    }


def fetch_and_save(track_ids: list, tracks_meta: dict, sp, writer, existing: set, genre_hint: str = "") -> int:
    added = 0
    for i in range(0, len(track_ids), 100):
        batch_ids  = track_ids[i:i+100]
        try:
            features = sp.audio_features(batch_ids)
        except Exception as e:
            print(f"  audio_features error: {e}")
            continue
        for af in features:
            if af is None: continue
            tid = af["id"]
            if tid in existing: continue
            meta = tracks_meta.get(tid, {})
            nf   = normalize_features(af)
            row  = {
                "track_id":   tid,
                "name":       meta.get("name",""),
                "artist":     meta.get("artist",""),
                "album":      meta.get("album",""),
                "year":       meta.get("year",""),
                "genre_hint": genre_hint,
                "popularity": meta.get("popularity",0),
                "duration_ms":af.get("duration_ms",0),
                **nf,
            }
            writer.writerow(row)
            existing.add(tid)
            added += 1
    return added


def run(count: int = 2000, playlist_id: str = None, search_query: str = None, limit: int = 200):
    sp = get_sp()

    existing = set()
    write_mode = "a" if OUT_CSV.exists() else "w"
    if OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(row["track_id"])
        print(f"Existing songs: {len(existing)}")

    total_added = 0

    with open(OUT_CSV, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_mode == "w":
            writer.writeheader()

        # Single playlist mode
        if playlist_id:
            print(f"Fetching playlist: {playlist_id}")
            results = sp.playlist_tracks(playlist_id, limit=100)
            track_ids, meta = [], {}
            while results:
                for item in results["items"]:
                    t = item.get("track")
                    if t and t.get("id"):
                        track_ids.append(t["id"])
                        meta[t["id"]] = {
                            "name": t["name"], "artist": t["artists"][0]["name"],
                            "album": t["album"]["name"],
                            "year": t["album"]["release_date"][:4],
                            "popularity": t.get("popularity", 0),
                        }
                if results.get("next"):
                    results = sp.next(results)
                else:
                    break
            added = fetch_and_save(track_ids, meta, sp, writer, existing)
            print(f"  Added {added} songs from playlist")
            return

        # Single search query mode
        if search_query:
            queries = [search_query]
        else:
            queries = SEARCH_QUERIES

        for query in queries:
            if total_added >= count:
                break
            print(f"Searching: {query}")
            try:
                offset = 0
                for _ in range(limit // 50):
                    res = sp.search(q=query, type="track", limit=50, offset=offset, market="US")
                    items = res.get("tracks", {}).get("items", [])
                    if not items: break
                    track_ids, meta = [], {}
                    for t in items:
                        if t and t.get("id"):
                            track_ids.append(t["id"])
                            meta[t["id"]] = {
                                "name": t["name"], "artist": t["artists"][0]["name"],
                                "album": t["album"]["name"],
                                "year": t["album"]["release_date"][:4],
                                "popularity": t.get("popularity", 0),
                            }
                    added = fetch_and_save(track_ids, meta, sp, writer, existing, genre_hint=query)
                    total_added += added
                    offset += 50
                    time.sleep(0.1)
            except Exception as e:
                print(f"  Error on query '{query}': {e}")
            f.flush()

        # Also sweep curated playlists
        for pid in PLAYLIST_IDS:
            if total_added >= count:
                break
            try:
                print(f"Playlist: {pid}")
                results = sp.playlist_tracks(pid, limit=100)
                track_ids, meta = [], {}
                while results and len(track_ids) < 200:
                    for item in results["items"]:
                        t = item.get("track")
                        if t and t.get("id"):
                            track_ids.append(t["id"])
                            meta[t["id"]] = {
                                "name": t["name"], "artist": t["artists"][0]["name"],
                                "album": t["album"]["name"],
                                "year": t["album"]["release_date"][:4],
                                "popularity": t.get("popularity", 0),
                            }
                    results = sp.next(results) if results.get("next") else None
                added = fetch_and_save(track_ids, meta, sp, writer, existing)
                total_added += added
                time.sleep(0.2)
            except Exception as e:
                print(f"  Error on playlist {pid}: {e}")

    print(f"\nDone. Total new songs added: {total_added} -> {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",    type=int, default=2000, help="Target song count")
    parser.add_argument("--playlist", type=str, default=None, help="Spotify playlist ID")
    parser.add_argument("--search",   type=str, default=None, help="Single search query")
    parser.add_argument("--limit",    type=int, default=200,  help="Results per query")
    args = parser.parse_args()
    run(count=args.count, playlist_id=args.playlist, search_query=args.search, limit=args.limit)
