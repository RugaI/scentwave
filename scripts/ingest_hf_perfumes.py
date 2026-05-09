# -*- coding: utf-8 -*-
"""
Ingest perfumes from HuggingFace dataset into fragrances.csv.
Source: abhirajeshbhai/perfume_recommendation_llm (2104 perfumes)

Run: python scripts/ingest_hf_perfumes.py
"""

import sys, os, re, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np

BASE_DIR  = Path(__file__).parent.parent
NOTES_JSON = BASE_DIR / "data" / "note_families.json"
FRAGS_CSV  = BASE_DIR / "data" / "fragrances.csv"

# ── Load canonical notes ───────────────────────────────────────────────────
with open(NOTES_JSON, encoding="utf-8") as f:
    families_data = json.load(f)

pyramid_layer   = families_data["pyramid_layer"]
emotion_profile = families_data["emotion_profile"]
all_families    = families_data["families"]

# Build lookup tables
NOTE_TO_FAMILY = {}
NOTE_TO_LAYER  = {}
for fam, notes in all_families.items():
    for n in notes:
        NOTE_TO_FAMILY[n] = fam
for layer, notes in pyramid_layer.items():
    for n in notes:
        NOTE_TO_LAYER[n] = layer

CANONICAL = set(NOTE_TO_FAMILY.keys())

# Family → dominant VAD (used when note can't be resolved)
FAMILY_VAD = {
    "citrus":     (0.85, 0.75, 0.35),
    "floral":     (0.75, 0.55, 0.40),
    "woody":      (0.45, 0.35, 0.60),
    "oriental":   (0.55, 0.50, 0.65),
    "fresh":      (0.80, 0.70, 0.35),
    "gourmand":   (0.70, 0.45, 0.40),
    "chypre":     (0.55, 0.45, 0.60),
    "fougere":    (0.60, 0.50, 0.55),
    "green":      (0.70, 0.65, 0.40),
    "leather":    (0.40, 0.55, 0.75),
}


def normalize_note(s: str) -> str:
    s = s.lower().strip().rstrip(".")
    s = re.sub(r"[''`]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


# Explicit alias table  (raw normalized form → canonical)
ALIASES = {
    "agarwood": "oud", "oudh": "oud", "oud_wood": "oud", "dehn_al_oud": "oud",
    "myrrh": "frankincense", "olibanum": "frankincense", "incense": "frankincense",
    "ambergris": "ambroxan", "ambrette": "ambroxan", "fixative": "ambroxan",
    "musk": "white_musk", "musks": "white_musk", "white_musks": "white_musk",
    "clean_musk": "clean_musk", "musky": "white_musk",
    "vanilla_bean": "vanilla", "vanilla_extract": "vanilla", "vanillin": "vanilla",
    "peru_balsam": "benzoin", "tolu_balsam": "benzoin", "balsam": "benzoin",
    "resin": "benzoin", "resinous_notes": "benzoin", "labdanum_resin": "labdanum",
    "lavender": "geranium", "lavandin": "geranium",
    "ylang-ylang": "ylang_ylang", "ylang": "ylang_ylang",
    "lily_of_valley": "lily_of_the_valley", "muguet": "lily_of_the_valley",
    "pink_peppercorn": "pink_pepper", "red_pepper": "pink_pepper",
    "black_peppercorn": "black_pepper", "pepper": "black_pepper",
    "green_pepper": "black_pepper",
    "spearmint": "mint", "peppermint": "mint",
    "eucalyptus_leaf": "eucalyptus",
    "holy_basil": "basil", "thai_basil": "basil",
    "ginger_root": "ginger", "fresh_ginger": "ginger",
    "mandarin_orange": "mandarin", "blood_orange": "orange",
    "clementine": "mandarin", "tangerine": "mandarin",
    "pomelo": "grapefruit",
    "yuzu_zest": "yuzu", "green_yuzu": "yuzu",
    "petitgrain_bigarade": "petitgrain",
    "neroli_bigarade": "neroli",
    "rose_de_mai": "rose", "rose_absolute": "rose", "bulgarian_rose": "rose",
    "turkish_rose": "rose", "centifolia_rose": "rose", "damask_rose": "rose",
    "jasmine_absolute": "jasmine", "sambac_jasmine": "jasmine",
    "geranium_bourbon": "geranium", "rose_geranium": "geranium",
    "orris_root": "iris", "iris_pallida": "iris", "orris": "iris",
    "violet_leaf": "violet", "violet_flower": "violet",
    "peony_blossom": "peony",
    "magnolia_blossom": "magnolia",
    "tuberose_absolute": "tuberose",
    "cardamom_seed": "cardamom",
    "cinnamon_bark": "cinnamon", "cinnamon_leaf": "cinnamon",
    "saffron_thread": "saffron", "saffron_absolute": "saffron",
    "clove_bud": "clove", "clove_leaf": "clove",
    "coffee_bean": "coffee", "roasted_coffee": "coffee", "espresso": "coffee",
    "black_coffee": "coffee",
    "guaiac": "guaiac_wood", "guaiacwood": "guaiac_wood",
    "scots_pine": "pine", "pine_needle": "pine",
    "oakmoss_absolute": "oakmoss", "tree_moss": "oakmoss",
    "sea_salt": "marine", "aquatic": "marine", "ocean": "marine",
    "water": "marine", "sea": "marine", "marine_notes": "marine",
    "fresh_water": "marine", "sea_spray": "marine",
    "cucumber_skin": "cucumber",
    "green_tea_leaf": "green_tea",
    "sandalwood_mysore": "sandalwood", "australian_sandalwood": "sandalwood",
    "amyris": "sandalwood",
    "cedarwood": "cedar", "atlas_cedar": "cedar", "virginia_cedar": "cedar",
    "himalayan_cedar": "cedar",
    "vetiver_root": "vetiver", "haitian_vetiver": "vetiver",
    "dark_patchouli": "patchouli", "patchouli_heart": "patchouli",
    "birch_tar": "birch", "birch_leaf": "birch",
    "amber_wood": "amber", "warm_amber": "amber", "solar_amber": "amber",
    "benzoin_resin": "benzoin", "siam_benzoin": "benzoin",
    "tonka": "tonka_bean", "coumarin": "tonka_bean",
    "labdanum_absolute": "labdanum", "cistus": "labdanum",
    "frankincense_resin": "frankincense", "boswellia": "frankincense",
    "iso_e": "iso_e_super",
    "leather_accord": "leather", "russian_leather": "leather",
    "suede_accord": "suede",
    "virginia_tobacco": "tobacco", "dark_tobacco": "tobacco",
    "almond_blossom": "almond", "bitter_almond": "almond",
    "honey_beeswax": "honey", "beeswax": "honey",
    "coconut_milk": "coconut",
    "caramel_accord": "caramel", "salted_caramel": "caramel",
    # Generic category fallbacks
    "wood": "cedar", "woods": "cedar", "woody_notes": "cedar",
    "floral": "rose", "floral_notes": "rose", "flowers": "rose",
    "spice": "black_pepper", "spices": "black_pepper",
    "citrus": "bergamot", "citrus_notes": "bergamot",
    "fruity": "mandarin", "fruit": "mandarin",
    "sweet": "vanilla", "sweetness": "vanilla",
    "fresh": "bergamot", "freshness": "bergamot",
    "green": "green_tea", "green_notes": "green_tea",
    "earthy": "vetiver", "earth": "vetiver",
    "smoky": "tobacco", "smoke": "tobacco",
    "powdery": "iris",
    "herbal": "basil",
    "peach": "mandarin", "apple": "mandarin", "pear": "mandarin",
    "cherry": "mandarin", "plum": "mandarin", "apricot": "mandarin",
    "berry": "mandarin", "strawberry": "mandarin", "raspberry": "mandarin",
    "blackberry": "mandarin", "blueberry": "mandarin",
    "watermelon": "cucumber", "melon": "cucumber",
    "fig": "green_tea", "fig_leaf": "green_tea",
    "rhubarb": "green_tea", "blackcurrant": "mandarin",
    "chocolate": "coffee", "cocoa": "coffee",
    "praline": "caramel", "nougat": "caramel",
    "cream": "vanilla", "milk": "vanilla", "butter": "vanilla",
    "iris_wood": "iris",
}


def map_note(raw: str) -> str | None:
    n = normalize_note(raw)
    if n in CANONICAL:
        return n
    if n in ALIASES:
        return ALIASES[n]
    # Partial match: canonical note name contained in raw or vice versa
    for cn in CANONICAL:
        if cn in n or (len(n) > 3 and n in cn):
            return cn
    # Check aliases with partial match
    for alias, target in ALIASES.items():
        if alias in n or (len(alias) > 4 and n in alias):
            return target
    return None


def compute_vad_from_notes(notes: list) -> tuple:
    vals, arousals, doms = [], [], []
    for note in notes:
        ep = emotion_profile.get(note)
        if ep:
            vals.append(ep["valence"])
            arousals.append(ep["arousal"])
            doms.append(ep["dominance"])
    if not vals:
        return 0.5, 0.5, 0.5
    return float(np.mean(vals)), float(np.mean(arousals)), float(np.mean(doms))


def infer_family(notes: list) -> str:
    family_counts = {}
    for n in notes:
        fam = NOTE_TO_FAMILY.get(n, "unknown")
        family_counts[fam] = family_counts.get(fam, 0) + 1
    if not family_counts:
        return "floral"
    return max(family_counts, key=family_counts.get)


def assign_pyramid(notes: list) -> tuple:
    top, mid, base = [], [], []
    for n in notes:
        layer = NOTE_TO_LAYER.get(n, "middle")
        if layer == "top":
            top.append(n)
        elif layer == "base":
            base.append(n)
        else:
            mid.append(n)
    return top, mid, base


# ── Parse HuggingFace dataset ─────────────────────────────────────────────
def parse_hf_dataset() -> list:
    from datasets import load_dataset
    import warnings
    warnings.filterwarnings("ignore")

    ds = load_dataset("abhirajeshbhai/perfume_recommendation_llm", split="train")
    print(f"  HF dataset rows: {len(ds)}")

    perfumes = {}
    for row in ds:
        text = row["text"]
        m = re.search(r"ingredients:(.*?)###\s*Assistant:\s*(.+)", text)
        if not m:
            continue
        notes_str = m.group(1).strip()
        name = m.group(2).strip()
        raw_notes = [n.strip() for n in notes_str.split(",") if n.strip()]
        if name not in perfumes:
            perfumes[name] = raw_notes

    return list(perfumes.items())


# ── Main ingestion ─────────────────────────────────────────────────────────
def main():
    # Load existing perfumes
    existing_names = set()
    existing_rows  = []
    max_id = 0
    if FRAGS_CSV.exists():
        with open(FRAGS_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_names.add(row["name"].lower())
                existing_rows.append(row)
                try:
                    max_id = max(max_id, int(row["id"]))
                except (ValueError, KeyError):
                    pass
    print(f"Existing fragrances: {len(existing_rows)}")

    hf_perfumes = parse_hf_dataset()
    print(f"  Parsed {len(hf_perfumes)} unique perfumes")

    new_rows = []
    unmapped_notes = {}
    next_id = max_id + 1

    for name, raw_notes in hf_perfumes:
        if name.lower() in existing_names:
            continue

        # Map notes
        mapped = []
        for rn in raw_notes:
            cn = map_note(rn)
            if cn:
                mapped.append(cn)
            else:
                unmapped_notes[rn] = unmapped_notes.get(rn, 0) + 1

        if not mapped:
            mapped = ["rose"]  # default fallback

        # Deduplicate
        seen = set()
        mapped_dedup = []
        for n in mapped:
            if n not in seen:
                seen.add(n)
                mapped_dedup.append(n)
        mapped = mapped_dedup

        top_notes, mid_notes, base_notes = assign_pyramid(mapped)
        valence, arousal, dominance = compute_vad_from_notes(mapped)
        family = infer_family(mapped)

        # Mood tags from VAD
        mood_tags = []
        if valence > 0.65:
            mood_tags.append("joyful")
        elif valence < 0.40:
            mood_tags.append("melancholic")
        if arousal > 0.65:
            mood_tags.append("energetic")
        elif arousal < 0.40:
            mood_tags.append("calming")
        if dominance > 0.60:
            mood_tags.append("powerful")
        elif dominance < 0.40:
            mood_tags.append("delicate")
        if not mood_tags:
            mood_tags = ["balanced"]

        new_rows.append({
            "id":           next_id,
            "name":         name,
            "brand":        "Unknown",
            "family":       family,
            "top_notes":    "|".join(top_notes),
            "middle_notes": "|".join(mid_notes),
            "base_notes":   "|".join(base_notes),
            "mood_tags":    "|".join(mood_tags),
            "valence":      round(valence, 4),
            "arousal":      round(arousal, 4),
            "dominance":    round(dominance, 4),
            "source":       "huggingface",
        })
        next_id += 1

    print(f"  New perfumes to add: {len(new_rows)}")

    # Top unmapped notes
    if unmapped_notes:
        top_unmapped = sorted(unmapped_notes.items(), key=lambda x: -x[1])[:20]
        print("  Top unmapped notes:", [(k, v) for k, v in top_unmapped])

    # Write combined CSV
    all_rows = existing_rows + new_rows
    fieldnames = ["id","name","brand","family","top_notes","middle_notes",
                  "base_notes","mood_tags","valence","arousal","dominance","source"]

    with open(FRAGS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone: {len(all_rows)} total fragrances in {FRAGS_CSV}")
    print(f"  (+{len(new_rows)} new from HuggingFace)")


if __name__ == "__main__":
    main()
