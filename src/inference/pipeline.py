# -*- coding: utf-8 -*-
"""
ScentWave Advanced Inference Pipeline

Provides:
  1. run()           — single song → emotion + real matches + novel formula
  2. timeline()      — song segmented into 8 parts → scent arc over time
  3. playlist()      — multiple songs → a named scent collection
  4. reverse()       — perfume name → recommended songs
  5. scent_card()    — returns structured data for the shareable card
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

from src.models.scentwave   import ScentWave
from src.data.fragrance_db  import FragranceDB, get_db, CANONICAL_NOTES, NUM_NOTES
from src.data.features      import extract_features, PRESET_SONGS


# ── Luxury editorial description engine ─────────────────────────────────────

_OPENING_PHRASES = [
    "An olfactory portrait of", "The liquid embodiment of", "Distilled from the soul of",
    "Born where music meets memory —", "A perfume conjured by", "The invisible aura of",
    "Capturing the ineffable quality of", "Where sound becomes sensation —",
]
_VALENCE_ADJ = [
    ("melancholic", "shadowed"), ("bittersweet", "smouldering"),
    ("contemplative", "smoky"), ("balanced", "grounded"),
    ("warm", "sun-dappled"), ("luminous", "radiant"), ("euphoric", "effervescent"),
]
_AROUSAL_ADJ = [
    ("whispered", "still"),    ("hushed", "languid"),    ("measured", "unhurried"),
    ("breathing", "alive"),    ("animated", "pulsing"),  ("electric", "crackling"),
    ("explosive", "untamed"),
]
_DOMINANCE_ADJ = [
    ("intimate", "skin-close"),   ("soft", "powdery"),     ("gentle", "yielding"),
    ("confident", "assured"),     ("bold", "declarative"), ("powerful", "commanding"),
    ("absolute", "sovereign"),
]

_GENRE_SCENT_NARRATIVE = {
    "classical":  "with the restraint of a conservatory at dusk",
    "jazz":       "with the smoky intimacy of a late-night club",
    "pop":        "with the bright immediacy of a crowded dancefloor",
    "rock":       "with the raw voltage of amplifiers pushed to the edge",
    "hip-hop":    "with the urban cool of city streets after rain",
    "electronic": "with the crystalline precision of synthesised light",
    "ambient":    "with the oceanic depth of a soundscape without horizon",
    "soul":       "with the deep warmth of a voice that has lived every word",
    "metal":      "with the volcanic intensity of distortion and darkness",
    "country":    "with the honest warmth of open roads and woodsmoke",
    "latin":      "with the sensual heat of bodies moving as one",
    "r&b":        "with the silken confidence of midnight velvet",
}

def _detect_genre(song_name: str) -> str:
    name = song_name.lower()
    genre_keywords = {
        "classical": ["sonata","symphony","nocturne","canon","concerto","beethoven","chopin","mozart","bach","vivaldi","debussy","ludovico","april"],
        "jazz":      ["jazz","swing","bebop","miles","coltrane","bossa","samba","blues","take five","autumn leaves","fly me"],
        "hip-hop":   ["drake","kendrick","eminem","gambino","rap","hip hop","gang","trap"],
        "electronic":["daft","deadmau","avicii","garrix","edm","techno","house","trance","strobe","levels","sandstorm"],
        "ambient":   ["lofi","chill","ambient","weightless","marconi","experience","aphex"],
        "metal":     ["metallica","sabbath","slayer","megadeth","dragonforce","master","paranoid","fire flames"],
        "country":   ["dolly","kenny rogers","country","jolene","gambler","country roads"],
        "soul":      ["stevie","wonder","wonderful world","hallelujah","nina simone","bb king","thrill"],
        "latin":     ["despacito","bamba","oye","salsa","reggaeton","latin"],
        "rock":      ["queen","nirvana","led","rolling","guns","november","paint","hotel california","bohemian","stairway","sweet child","wish you were"],
        "r&b":       ["redbone","r&b","rnb","bitch better","no role"],
        "pop":       ["blinding","shape of you","bad guy","someone like you","uptown","perfect","levitating","watermelon","as it was"],
    }
    for genre, keywords in genre_keywords.items():
        for kw in keywords:
            if kw in name:
                return genre
    return "pop"


def _luxury_description(
    song_name: str,
    emotion:   dict,
    top:       list,
    middle:    list,
    base:      list,
    family_profile: dict,
) -> str:
    import random
    rng = random.Random(abs(hash(song_name)) % (2**31))

    V = emotion["valence"]
    A = emotion["arousal"]
    D = emotion["dominance"]

    v_idx = min(int(V * 6.99), 6)
    a_idx = min(int(A * 6.99), 6)
    d_idx = min(int(D * 6.99), 6)

    v_adj1, v_adj2 = _VALENCE_ADJ[v_idx]
    a_adj1, a_adj2 = _AROUSAL_ADJ[a_idx]
    d_adj1, d_adj2 = _DOMINANCE_ADJ[d_idx]

    genre = _detect_genre(song_name)
    genre_phrase = _GENRE_SCENT_NARRATIVE.get(genre, "with a character all its own")

    opening   = rng.choice(_OPENING_PHRASES)
    dom_fam   = max(family_profile, key=family_profile.get) if family_profile else "woody"
    dom_fam_s = dom_fam.replace("_", " ")

    top_note    = top[0][0].lower()    if top    else "bergamot"
    mid_note    = middle[0][0].lower() if middle else "rose"
    base_note   = base[0][0].lower()   if base   else "sandalwood"
    top2_note   = top[1][0].lower()    if len(top) > 1    else top_note
    base2_note  = base[1][0].lower()   if len(base) > 1   else base_note

    lines = [
        f"{opening} \"{song_name}\" — {genre_phrase}.",
        f"",
        f"A {v_adj1}, {a_adj1} composition with a {d_adj1} character. "
        f"Fundamentally {dom_fam_s} in nature, yet lit from within by a {v_adj2}, {a_adj2} spirit.",
        f"",
        f"It opens with the {a_adj2} brightness of **{top_note}** and **{top2_note}**, "
        f"a sparkling fanfare that evaporates into the warmth of **{mid_note}** at the heart. "
        f"The drydown is {d_adj2} and unhurried — **{base_note}** and **{base2_note}** "
        f"anchor the composition with {d_adj1} resolve.",
        f"",
        f"This formula does not exist in any bottle. It was created for this song alone.",
    ]
    return "\n".join(lines)


# ── Timeline: scent arc over 8 song segments ────────────────────────────────

_SEGMENT_NAMES = [
    "Intro", "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Bridge", "Outro", "Fade"
]

def _segment_features(features: np.ndarray, segment: int) -> np.ndarray:
    """
    Simulate how a song's emotional profile shifts across 8 segments.
    Intro/outro: lower energy. Chorus: peak energy+valence. Bridge: harmonic shift.
    """
    f = features.copy()
    t = segment / 7.0  # 0 to 1

    # Energy arc: low → build → peak at chorus (seg 3) → decay
    if segment == 0:   energy_mult = 0.55
    elif segment == 1: energy_mult = 0.75
    elif segment == 2: energy_mult = 0.88
    elif segment == 3: energy_mult = 1.10
    elif segment == 4: energy_mult = 0.82
    elif segment == 5: energy_mult = 0.95
    elif segment == 6: energy_mult = 0.70
    else:              energy_mult = 0.45

    valence_shift  = np.sin(t * np.pi) * 0.15   # peaks in middle
    arousal_shift  = (energy_mult - 1.0) * 0.3

    f[1]  = np.clip(f[1]  * energy_mult, 0, 1)   # energy
    f[9]  = np.clip(f[9]  + valence_shift, 0, 1)  # valence
    f[10] = np.clip(f[10] * energy_mult, 0, 1)    # tempo_norm

    return f


def timeline(model: ScentWave, features: np.ndarray, song_name: str) -> List[Dict]:
    """
    Returns 8 segments, each with emotion + dominant notes.
    """
    segments = []
    for i, name in enumerate(_SEGMENT_NAMES):
        seg_feat = _segment_features(features, i)
        result   = model.predict(seg_feat, top_k=1, note_threshold=0.015)
        segments.append({
            "segment":      name,
            "index":        i,
            "emotion":      result["emotion"],
            "top_notes":    result["generated"]["top"][:3],
            "base_notes":   result["generated"]["base"][:2],
            "family":       max(result["generated"]["family_profile"],
                               key=result["generated"]["family_profile"].get,
                               default="unknown"),
        })
    return segments


# ── Playlist: multiple songs → a named collection ────────────────────────────

_COLLECTION_NAMES = [
    "The {adj} Archive", "Accord {adj}", "Suite {adj}", "{adj} Memories",
    "The {adj} Hours", "Notes on {adj}", "Elixir {adj}",
]
_COLLECTION_ADJ = {
    "high_V_low_A":  ["Serene", "Luminous", "Tranquil", "Pastoral"],
    "high_V_high_A": ["Radiant", "Vivid", "Solar", "Effervescent"],
    "low_V_low_A":   ["Nocturnal", "Shadowed", "Intimate", "Hushed"],
    "low_V_high_A":  ["Tempestuous", "Electric", "Raw", "Dark"],
    "balanced":      ["Signature", "Timeless", "Universal", "Complete"],
}

def playlist(model: ScentWave,
             songs: List[Tuple[str, str]],   # [(name, artist), ...]
             ) -> Dict:
    """Analyzes multiple songs and builds a named scent collection."""
    import random
    results = []
    all_emotions = []
    all_formulas = np.zeros(NUM_NOTES)

    for song_name, artist in songs:
        features, meta = extract_features(song_name, artist)
        result = model.predict(features, top_k=1)
        result["song_name"] = meta["name"]
        result["artist"]    = meta["artist"]
        results.append(result)

        e = result["emotion"]
        all_emotions.append([e["valence"], e["arousal"], e["dominance"]])

    all_emotions  = np.array(all_emotions)
    mean_emotion  = all_emotions.mean(axis=0)
    V, A, D       = mean_emotion

    # Classify collection mood
    if V > 0.6 and A < 0.5:   mood_key = "high_V_low_A"
    elif V > 0.6 and A >= 0.5: mood_key = "high_V_high_A"
    elif V <= 0.4 and A < 0.5: mood_key = "low_V_low_A"
    elif V <= 0.4 and A >= 0.5:mood_key = "low_V_high_A"
    else:                       mood_key = "balanced"

    rng  = random.Random(hash(tuple(songs[0])) % (2**31))
    adj  = rng.choice(_COLLECTION_ADJ[mood_key])
    tmpl = rng.choice(_COLLECTION_NAMES)
    name = tmpl.format(adj=adj)

    return {
        "collection_name": name,
        "mean_emotion": {
            "valence":   round(float(V), 3),
            "arousal":   round(float(A), 3),
            "dominance": round(float(D), 3),
        },
        "songs": results,
        "description": (
            f'"{name}" is a collection of {len(songs)} accords, each born from a different song. '
            f"United by a {['shadowed','balanced','luminous'][1 if 0.4<V<0.6 else (2 if V>=0.6 else 0)]} emotional thread "
            f"and a {['still','moderate','electric'][1 if 0.4<A<0.6 else (2 if A>=0.6 else 0)]} energy, "
            f"they form a complete olfactory narrative from opening to close."
        ),
    }


# ── Reverse: perfume → recommended songs ─────────────────────────────────────

def reverse(db: FragranceDB, perfume_name: str, top_k: int = 5) -> Dict:
    """
    Given a perfume name, finds the closest songs from the preset library
    by matching VAD vectors.
    """
    # Find perfume in DB
    match = None
    name_lower = perfume_name.lower().strip()
    for r in db.records:
        if name_lower in r["name"].lower():
            match = r
            break

    if match is None:
        return {"error": f"Perfume '{perfume_name}' not found in database."}

    perf_vad = np.array([float(match["valence"]), float(match["arousal"]),
                         float(match["dominance"])], dtype=np.float32)

    # Compute VAD for all preset songs
    song_vads = {}
    for song_key, feat_list in PRESET_SONGS.items():
        feat = np.array(feat_list, dtype=np.float32)
        V = feat[9]
        A = (feat[1] + feat[10]) / 2
        D = (feat[3] + feat[4]) / 2
        song_vads[song_key] = np.array([V, A, D])

    # Cosine similarity
    perf_norm = perf_vad / (np.linalg.norm(perf_vad) + 1e-8)
    scored = []
    for song, vad in song_vads.items():
        v_norm = vad / (np.linalg.norm(vad) + 1e-8)
        sim = float(np.dot(perf_norm, v_norm))
        scored.append((song, sim))
    scored.sort(key=lambda x: -x[1])

    top_songs = scored[:top_k]
    top_score = top_songs[0][1]

    return {
        "perfume": {
            "name":      match["name"],
            "brand":     match["brand"],
            "family":    match["family"],
            "top_notes": match["top_notes"],
        },
        "recommended_songs": [
            {
                "song":      song.title(),
                "match_pct": round(100.0 - (top_score - sim) / max(top_score - scored[-1][1], 1e-6) * 25, 1),
            }
            for song, sim in top_songs
        ],
        "description": (
            f"{match['name']} by {match['brand']} shares its emotional signature with "
            f"these songs — they occupy the same space in the landscape of feeling."
        ),
    }


# ── Scent card data (for shareable visual) ────────────────────────────────────

def scent_card(song_name: str, result: Dict) -> Dict:
    """Returns structured data to render a shareable luxury scent card."""
    top3    = result["generated"]["top"][:3]
    base2   = result["generated"]["base"][:2]
    dom_fam = max(result["generated"]["family_profile"],
                  key=result["generated"]["family_profile"].get,
                  default="woody")
    e = result["emotion"]

    intensity_word = (
        "Intense"   if e["dominance"] > 0.65 else
        "Moderate"  if e["dominance"] > 0.40 else
        "Delicate"
    )
    mood_word = (
        "Joyful"      if e["valence"] > 0.70 and e["arousal"] > 0.60 else
        "Euphoric"    if e["valence"] > 0.70 else
        "Melancholic" if e["valence"] < 0.35 else
        "Powerful"    if e["dominance"] > 0.70 else
        "Balanced"
    )

    return {
        "song_name":   song_name,
        "formula_name": f"No. {abs(hash(song_name)) % 9999 + 1:04d}",
        "tagline":     f"A {mood_word.lower()}, {intensity_word.lower()} {dom_fam.replace('_',' ')} accord",
        "top_notes":   [n for n, _ in top3],
        "base_notes":  [n for n, _ in base2],
        "emotion":     e,
        "intensity":   intensity_word,
        "mood":        mood_word,
        "family":      dom_fam.replace("_", " ").title(),
        "best_match":  result["retrieved"][0]["name"] if result["retrieved"] else "—",
    }
