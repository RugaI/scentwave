# -*- coding: utf-8 -*-
"""
Loads fragrances.csv and note_families.json, exposes the fragrance database
as a structured object used by both the retrieval and generation heads.
"""

import json
import csv
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Tuple

BASE = Path(__file__).parent.parent.parent

# All canonical note names (must match note_families.json keys in order)
CANONICAL_NOTES: List[str] = [
    # Citrus
    "bergamot","lemon","orange","grapefruit","mandarin","lime","yuzu","petitgrain",
    # Floral
    "rose","jasmine","ylang_ylang","peony","iris","violet","geranium","neroli",
    "tuberose","lily_of_the_valley","magnolia",
    "osmanthus","gardenia","carnation","freesia","heliotrope",
    # Woody
    "sandalwood","cedar","vetiver","oud","patchouli","guaiac_wood","birch","pine","oakmoss",
    "cypress","juniper",
    # Oriental
    "vanilla","amber","benzoin","tonka_bean","labdanum","frankincense","styrax","elemi",
    # Musk
    "white_musk","clean_musk","black_musk","ambroxan","iso_e_super",
    "cashmeran","civet","castoreum",
    # Fresh
    "marine","green_tea","cucumber","mint","basil","eucalyptus","galbanum",
    # Spicy
    "black_pepper","pink_pepper","cinnamon","cardamom","ginger","clove","saffron",
    "nutmeg","coriander","cumin","clary_sage",
    # Gourmand
    "caramel","coffee","almond","honey","coconut",
    # Leather
    "leather","suede","tobacco",
    # Aromatic
    "davana",
]

NOTE_INDEX: Dict[str, int] = {n: i for i, n in enumerate(CANONICAL_NOTES)}
NUM_NOTES = len(CANONICAL_NOTES)  # 78


def _normalize_note(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def notes_to_vector(notes_str: str, weight: float = 1.0) -> np.ndarray:
    vec = np.zeros(NUM_NOTES, dtype=np.float32)
    for raw in notes_str.split(","):
        key = _normalize_note(raw)
        if key in NOTE_INDEX:
            vec[NOTE_INDEX[key]] += weight
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def pyramid_to_vector(top: str, middle: str, base: str) -> np.ndarray:
    """Weighted blend: top 20%, middle 35%, base 45%."""
    v  = notes_to_vector(top, 0.20)
    v += notes_to_vector(middle, 0.35)
    v += notes_to_vector(base, 0.45)
    total = v.sum()
    if total > 0:
        v /= total
    return v


class FragranceDB:
    def __init__(self, csv_path: str = None, families_path: str = None):
        csv_path     = csv_path     or str(BASE / "data" / "fragrances.csv")
        families_path= families_path or str(BASE / "data" / "note_families.json")

        with open(families_path, encoding="utf-8") as f:
            fam_data = json.load(f)
        self.note_emotions: Dict[str, dict] = fam_data["emotion_profile"]
        self.pyramid_layers: Dict[str, List[str]] = fam_data["pyramid_layer"]
        self.families: Dict[str, List[str]] = fam_data["families"]

        self.records: List[Dict] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.records.append(row)

        # Pre-compute note vectors and VAD arrays
        self.note_vectors  = np.stack([
            pyramid_to_vector(r["top_notes"], r["middle_notes"], r["base_notes"])
            for r in self.records
        ])                                                      # (N, 60)

        self.vad_vectors = np.array([
            [float(r["valence"]), float(r["arousal"]), float(r["dominance"])]
            for r in self.records
        ], dtype=np.float32)                                    # (N, 3)

    def __len__(self):
        return len(self.records)

    def get(self, idx: int) -> Dict:
        return self.records[idx]

    def search_by_vad(self, vad: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Cosine similarity in VAD space."""
        q = vad / (np.linalg.norm(vad) + 1e-8)
        db = self.vad_vectors / (np.linalg.norm(self.vad_vectors, axis=1, keepdims=True) + 1e-8)
        sims = db @ q
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idxs]

    def search_by_notes(self, note_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Cosine similarity in note-embedding space."""
        q = note_vec / (np.linalg.norm(note_vec) + 1e-8)
        db = self.note_vectors / (np.linalg.norm(self.note_vectors, axis=1, keepdims=True) + 1e-8)
        sims = db @ q
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idxs]

    def vad_from_audio_features(self, features: np.ndarray) -> np.ndarray:
        """
        Map 12-dim audio feature vector to 3-dim VAD.
        valence   = features[9]  (Spotify valence or mapped equivalent)
        arousal   = (energy + tempo_norm) / 2
        dominance = (loudness_norm + mode) / 2
        """
        valence   = float(features[9])
        arousal   = float((features[1] + features[10]) / 2)
        dominance = float((features[3] + features[4]) / 2)
        return np.array([valence, arousal, dominance], dtype=np.float32)

    def get_note_name(self, idx: int) -> str:
        return CANONICAL_NOTES[idx]

    def notes_in_layer(self, layer: str) -> List[int]:
        return [NOTE_INDEX[n] for n in self.pyramid_layers[layer] if n in NOTE_INDEX]

    def family_summary(self, note_vec: np.ndarray) -> Dict[str, float]:
        """Returns % contribution of each family."""
        summary = {}
        for fam, notes in self.families.items():
            idxs = [NOTE_INDEX[n] for n in notes if n in NOTE_INDEX]
            summary[fam] = float(note_vec[idxs].sum())
        total = sum(summary.values())
        if total > 0:
            summary = {k: v / total for k, v in summary.items()}
        return dict(sorted(summary.items(), key=lambda x: -x[1]))


@lru_cache(maxsize=1)
def get_db() -> FragranceDB:
    return FragranceDB()
