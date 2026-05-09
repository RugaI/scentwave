# -*- coding: utf-8 -*-
"""
Fetch Spotify songs using a Bearer token directly (no client credentials needed).
Run immediately — Bearer tokens expire in ~1 hour.

Usage: python scripts/fetch_with_token.py <bearer_token>
"""
import sys, os, csv, time, json, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SPOTIFY_BEARER_TOKEN", "")
if not TOKEN:
    print("Usage: python scripts/fetch_with_token.py <bearer_token>")
    sys.exit(1)

OUT_CSV = Path(__file__).parent.parent / "data" / "songs.csv"

FIELDNAMES = [
    "track_id","name","artist","album","year","genre_hint",
    "danceability","energy","key_norm","loudness_norm","mode",
    "speechiness","acousticness","instrumentalness","liveness",
    "valence","tempo_norm","time_sig_norm",
    "popularity","duration_ms",
]

SEARCH_QUERIES = [
    # Genres
    "genre:classical", "genre:jazz", "genre:blues", "genre:country",
    "genre:rock", "genre:metal", "genre:punk", "genre:indie",
    "genre:pop", "genre:r-n-b", "genre:hip-hop", "genre:rap",
    "genre:electronic", "genre:ambient", "genre:techno", "genre:house",
    "genre:latin", "genre:reggae", "genre:soul", "genre:funk",
    "genre:folk", "genre:alternative", "genre:grunge", "genre:disco",
    "genre:opera", "genre:flamenco", "genre:bossa-nova", "genre:gospel",
    # Decades
    "year:1960-1969", "year:1970-1979", "year:1980-1989",
    "year:1990-1999", "year:2000-2009", "year:2010-2019", "year:2020-2024",
    # Moods
    "mood sad", "mood happy", "mood dark", "mood romantic",
    "chill relaxing", "workout energy", "focus study", "late night",
    "rain melancholy", "summer vibes", "winter cozy", "party dance",
    # Instruments
    "piano solo", "guitar acoustic", "violin classical", "jazz trumpet",
    "drum bass electronic", "synthesizer ambient", "saxophone jazz",
    "orchestra symphony", "choir vocal", "bass guitar funk",
    # Artists/styles
    "beethoven symphony", "mozart piano", "bach fugue",
    "blues guitar delta", "jazz bebop", "soul motown",
    "hip hop beats", "trap music", "lo fi chill",
    "cinematic score", "video game music", "film soundtrack",
]

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
    "37i9dQZF1DWT7XSlIpX3s6",  # Dark & Stormy
    "37i9dQZF1DX3rxVfibe1L0",  # Mood Booster
    "37i9dQZF1DX9sIqqvKsjEJ",  # Songs to Sing in the Car
    "37i9dQZF1DXbITWG1ZJKYt",  # Blues Classics
    "37i9dQZF1DX0h0QnLkMBl4",  # Peaceful Guitar
    "37i9dQZF1DWWQRwui0ExPn",  # lofi beats
    "37i9dQZF1DX6GwdWRQMQpq",  # Soft Pop Hits
    "37i9dQZF1DX4fpCWaHOned",  # Latin Pop
    "37i9dQZF1DX8FwnYE6PRvL",  # Rock en Español
    "37i9dQZF1DX6VDO8a6cQME",  # Soul Classics
    "37i9dQZF1DX2SK4ytI2KAh",  # Funk Classics
    "37i9dQZF1DWTkIwO4Hs75O",  # Classical Essentials
    "37i9dQZF1DX4UtSsGT1Sbe",  # All Out 80s
    "37i9dQZF1DXbTxeAdrVG2l",  # All Out 90s
    "37i9dQZF1DX4o1oenSJRJd",  # All Out 00s
    "37i9dQZF1DX5Ejj0EkURtP",  # All Out 2010s
    "37i9dQZF1DX6R7QUWePReA",  # African Heat
    "37i9dQZF1DWYmmr74INQlb",  # Arab X
]


def api_get(endpoint: str) -> dict:
    req = urllib.request.Request(
        f"https://api.spotify.com/{endpoint}",
        headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            retry = int(e.headers.get("Retry-After", 3))
            print(f"  Rate limited — waiting {retry}s")
            time.sleep(retry)
            return api_get(endpoint)
        elif e.code == 401:
            print("  Token expired! Get a new one from developer.spotify.com")
            sys.exit(1)
        return {}
    except Exception as e:
        print(f"  Request error: {e}")
        return {}


def normalize_features(af: dict) -> dict:
    return {
        "danceability":     af.get("danceability", 0.5),
        "energy":           af.get("energy", 0.5),
        "key_norm":         af.get("key", 5) / 11.0,
        "loudness_norm":    min((af.get("loudness", -10) + 60.0) / 60.0, 1.0),
        "mode":             float(af.get("mode", 1)),
        "speechiness":      af.get("speechiness", 0.05),
        "acousticness":     af.get("acousticness", 0.3),
        "instrumentalness": af.get("instrumentalness", 0.1),
        "liveness":         af.get("liveness", 0.15),
        "valence":          af.get("valence", 0.5),
        "tempo_norm":       min(af.get("tempo", 120) / 220.0, 1.0),
        "time_sig_norm":    (af.get("time_signature", 4) - 1) / 6.0,
    }


def fetch_audio_features(track_ids: list, tracks_meta: dict,
                          writer, existing: set, genre_hint: str = "") -> int:
    added = 0
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        ids_param = ",".join(batch)
        data = api_get(f"v1/audio-features?ids={ids_param}")
        features_list = data.get("audio_features", [])
        for af in features_list:
            if af is None:
                continue
            tid = af.get("id")
            if not tid or tid in existing:
                continue
            meta = tracks_meta.get(tid, {})
            nf = normalize_features(af)
            row = {
                "track_id":    tid,
                "name":        meta.get("name", ""),
                "artist":      meta.get("artist", ""),
                "album":       meta.get("album", ""),
                "year":        meta.get("year", ""),
                "genre_hint":  genre_hint,
                "popularity":  meta.get("popularity", 0),
                "duration_ms": af.get("duration_ms", 0),
                **nf,
            }
            writer.writerow(row)
            existing.add(tid)
            added += 1
        time.sleep(0.05)
    return added


def run():
    existing = set()
    write_mode = "a" if OUT_CSV.exists() else "w"
    if OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(row["track_id"])
        print(f"Existing songs in CSV: {len(existing)}")

    total_added = 0

    with open(OUT_CSV, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_mode == "w":
            writer.writeheader()

        # 1. User's own top tracks (all time ranges)
        for range_ in ["short_term", "medium_term", "long_term"]:
            data = api_get(f"v1/me/top/tracks?time_range={range_}&limit=50")
            items = data.get("items", [])
            track_ids, meta = [], {}
            for t in items:
                if t and t.get("id"):
                    track_ids.append(t["id"])
                    meta[t["id"]] = {
                        "name": t["name"],
                        "artist": t["artists"][0]["name"],
                        "album": t["album"]["name"],
                        "year": t["album"].get("release_date", "")[:4],
                        "popularity": t.get("popularity", 0),
                    }
            added = fetch_audio_features(track_ids, meta, writer, existing, genre_hint=f"user_top_{range_}")
            total_added += added
            print(f"  User top tracks ({range_}): +{added}")

        # 2. Search queries
        for query in SEARCH_QUERIES:
            print(f"Searching: {query} ...")
            track_ids, meta = [], {}
            for offset in [0, 50, 150, 250]:
                q = urllib.parse.quote(query)
                data = api_get(f"v1/search?q={q}&type=track&limit=50&offset={offset}&market=US")
                items = data.get("tracks", {}).get("items", [])
                if not items:
                    break
                for t in items:
                    if t and t.get("id") and t["id"] not in existing:
                        track_ids.append(t["id"])
                        meta[t["id"]] = {
                            "name": t["name"],
                            "artist": t["artists"][0]["name"],
                            "album": t["album"]["name"],
                            "year": t["album"].get("release_date", "")[:4],
                            "popularity": t.get("popularity", 0),
                        }
                time.sleep(0.1)
            added = fetch_audio_features(track_ids, meta, writer, existing, genre_hint=query)
            total_added += added
            print(f"  +{added} songs (total: {total_added})")
            f.flush()

        # 3. Playlists
        for pid in PLAYLIST_IDS:
            print(f"Playlist {pid} ...")
            track_ids, meta = [], {}
            endpoint = f"v1/playlists/{pid}/tracks?limit=100"
            while endpoint:
                data = api_get(endpoint)
                for item in data.get("items", []):
                    t = item.get("track")
                    if t and t.get("id") and t["id"] not in existing:
                        track_ids.append(t["id"])
                        meta[t["id"]] = {
                            "name": t["name"],
                            "artist": t["artists"][0]["name"] if t.get("artists") else "",
                            "album": t["album"]["name"] if t.get("album") else "",
                            "year": t["album"].get("release_date", "")[:4] if t.get("album") else "",
                            "popularity": t.get("popularity", 0),
                        }
                next_url = data.get("next")
                if next_url and len(track_ids) < 300:
                    endpoint = next_url.replace("https://api.spotify.com/", "")
                else:
                    break
                time.sleep(0.1)
            added = fetch_audio_features(track_ids, meta, writer, existing)
            total_added += added
            print(f"  +{added} (total: {total_added})")
            f.flush()

    print(f"\nDone. Total new songs: {total_added} -> {OUT_CSV}")


if __name__ == "__main__":
    run()
