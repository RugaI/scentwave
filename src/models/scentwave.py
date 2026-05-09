# -*- coding: utf-8 -*-
"""
ScentWave — unified model.

One forward pass → two outputs:
  1. Retrieved perfumes  (existing, real)
  2. Generated formula   (novel, never existed)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.models.encoder    import AudioEncoder
from src.models.retrieval  import RetrievalHead
from src.models.generator  import GenerationHead
from src.data.fragrance_db import FragranceDB, get_db, NUM_NOTES


class ScentWave(nn.Module):
    def __init__(self,
                 input_dim:   int = 12,
                 hidden_dim:  int = 256,
                 latent_dim:  int = 512,
                 emotion_dim: int = 3,
                 embed_dim:   int = 128,
                 num_notes:   int = NUM_NOTES,
                 dropout:     float = 0.2):
        super().__init__()

        self.encoder   = AudioEncoder(input_dim, hidden_dim, latent_dim, emotion_dim, dropout)
        self.retrieval = RetrievalHead(latent_dim, emotion_dim, embed_dim, dropout)
        self.generator = GenerationHead(latent_dim, emotion_dim, num_notes, dropout)

        self._db: Optional[FragranceDB] = None

    def forward(self, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent, emotion = self.encoder(audio_features)
        gen_out = self.generator(latent, emotion)
        query   = self.retrieval(latent, emotion)
        return {
            "latent":    latent,
            "emotion":   emotion,
            "query_emb": query,
            **{f"gen_{k}": v for k, v in gen_out.items()},
        }

    # ── Convenience: build retrieval index from DB ───────────────────────────
    def init_db(self, db: FragranceDB = None):
        self._db = db or get_db()
        device = next(self.parameters()).device
        self.retrieval.build_index(self._db.vad_vectors, device=device)

    # ── Full inference pipeline ──────────────────────────────────────────────
    @torch.no_grad()
    def predict(self,
                audio_features: np.ndarray,
                top_k: int = 3,
                note_threshold: float = 0.02) -> Dict:
        """
        Input:  12-dim numpy feature vector
        Output: {
            "emotion": [V, A, D],
            "retrieved": [{"rank", "name", "brand", "match_pct", "family",
                           "top_notes", "middle_notes", "base_notes", "mood_tags"}, ...],
            "generated": {"top": [(note, pct)], "middle": [...], "base": [...],
                          "description": str, "family_profile": {...}},
        }
        """
        if self._db is None:
            self.init_db()

        x = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0)
        device = next(self.parameters()).device
        x = x.to(device)

        latent, emotion = self.encoder(x)

        # ── Retrieval ────────────────────────────────────────────────────────
        matches = self.retrieval.retrieve(latent, emotion, top_k=top_k)
        # Re-scale scores: top match = 100%, rest relative to top
        if matches:
            top_score = max(s for _, s in matches)
            min_score = min(s for _, s in matches)
            score_range = max(top_score - min_score, 1e-6)

        retrieved = []
        for rank, (idx, score) in enumerate(matches, 1):
            r = self._db.get(idx)
            # Map to [75, 100] range — meaningful, not inflated
            pct = 100.0 - (top_score - score) / score_range * 25.0
            retrieved.append({
                "rank":         rank,
                "name":         r["name"],
                "brand":        r["brand"],
                "match_pct":    round(pct, 1),
                "family":       r["family"],
                "top_notes":    r["top_notes"],
                "middle_notes": r["middle_notes"],
                "base_notes":   r["base_notes"],
                "mood_tags":    r["mood_tags"],
            })

        # ── Generation ───────────────────────────────────────────────────────
        gen_formula = self.generator.generate(latent, emotion, threshold=note_threshold)
        family_profile = self._db.family_summary(gen_formula["full"])

        description = _describe_formula(
            emotion.cpu().numpy()[0],
            gen_formula["top"],
            gen_formula["middle"],
            gen_formula["base"],
            family_profile,
        )

        return {
            "emotion":   {
                "valence":   round(float(emotion[0, 0]), 3),
                "arousal":   round(float(emotion[0, 1]), 3),
                "dominance": round(float(emotion[0, 2]), 3),
            },
            "retrieved": retrieved,
            "generated": {
                "top":            gen_formula["top"],
                "middle":         gen_formula["middle"],
                "base":           gen_formula["base"],
                "description":    description,
                "family_profile": {k: round(v * 100, 1) for k, v in family_profile.items()},
            },
        }

    # ── Save / load ──────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "ScentWave":
        data  = torch.load(path, map_location="cpu")
        sd    = {k: v for k, v in data["state_dict"].items()
                 if "perfume_embeddings" not in k}
        model = cls(**kwargs)
        model.load_state_dict(sd, strict=False)
        return model


# ── Scent description generator ─────────────────────────────────────────────

_VALENCE_WORDS  = ["melancholic","bittersweet","contemplative","balanced","uplifting","euphoric","radiant"]
_AROUSAL_WORDS  = ["serene","tranquil","calm","moderate","vibrant","energetic","electric"]
_DOMINANCE_WORDS= ["delicate","soft","gentle","grounded","confident","bold","commanding"]

def _describe_formula(emotion: np.ndarray,
                      top: list, middle: list, base: list,
                      family_profile: dict) -> str:
    V, A, D = emotion
    v_word = _VALENCE_WORDS[min(int(V * 6.99), 6)]
    a_word = _AROUSAL_WORDS[min(int(A * 6.99), 6)]
    d_word = _DOMINANCE_WORDS[min(int(D * 6.99), 6)]

    dominant_family = max(family_profile, key=family_profile.get) if family_profile else "woody"
    top_note    = top[0][0]    if top    else "citrus"
    middle_note = middle[0][0] if middle else "floral"
    base_note   = base[0][0]   if base   else "musk"

    return (
        f"A {v_word}, {a_word} fragrance with a {d_word} character. "
        f"Opens with {top_note.lower()}, blooms into {middle_note.lower()}, "
        f"and settles on a {base_note.lower()} base. "
        f"Predominantly {dominant_family.replace('_', ' ')} in nature — "
        f"a scent that has never been bottled before."
    )
