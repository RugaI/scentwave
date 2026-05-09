# -*- coding: utf-8 -*-
"""
ScentWave — Fragrantica Scraper
Scrapes perfume data from Fragrantica and appends to data/fragrances.csv.

Usage:
    python scripts/scrape_fragrantica.py --pages 50   # ~1000 perfumes
    python scripts/scrape_fragrantica.py --pages 500  # ~10000 perfumes
    python scripts/scrape_fragrantica.py --url https://www.fragrantica.com/perfume/Dior/Sauvage-34187.html

IMPORTANT: Be respectful — adds a 1-2s delay between requests.
"""

import sys, os, csv, json, time, re, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
OUT_CSV     = BASE_DIR / "data" / "fragrances.csv"
FAMILIES_F  = BASE_DIR / "data" / "note_families.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

FIELDNAMES = ["id","name","brand","family","top_notes","middle_notes",
              "base_notes","mood_tags","valence","arousal","dominance","source"]

# Entry-point pages (designer lists, popular, etc.)
SEED_URLS = [
    "https://www.fragrantica.com/designers/",
    "https://www.fragrantica.com/news/",
    "https://www.fragrantica.com/perfume/",
]

# Popular designers to scrape
DESIGNER_SLUGS = [
    "Chanel","Dior","Guerlain","Hermes","Tom-Ford","Maison-Margiela",
    "Byredo","Le-Labo","Jo-Malone-London","Creed","Amouage","MFK",
    "Frederic-Malle","Serge-Lutens","Diptyque","Penhaligons","YSL",
    "Armani","Versace","Givenchy","Prada","Lancome","Calvin-Klein",
    "Dolce-Gabbana","Burberry","Marc-Jacobs","Paco-Rabanne","Valentino",
    "Gucci","Narciso-Rodriguez","Viktor-Rolf","Thierry-Mugler","Parfums-de-Marly",
    "Initio","Xerjoff","Kilian","Montale","Mancera","Bvlgari","Roja-Dove",
    "Lattafa","Rasasi","Afnan","Armaf","Atelier-Cologne","Acqua-di-Parma",
]


def _extract_notes_from_html(html: str):
    """Parse note pyramid from Fragrantica HTML (no JS)."""
    import html as html_lib

    top, middle, base = [], [], []

    # Notes appear in divs with class pyramid-level
    layers = re.findall(
        r'<div[^>]*class="[^"]*notes-box[^"]*"[^>]*>(.*?)</div>',
        html, re.DOTALL
    )
    # Fallback: look for "Top notes", "Middle notes", "Base notes" sections
    for section, label in [
        (top,    ["Top notes", "top notes", "Head notes"]),
        (middle, ["Middle notes", "Heart notes", "middle notes"]),
        (base,   ["Base notes", "base notes", "Bottom notes"]),
    ]:
        for lbl in label:
            pat = rf'{re.escape(lbl)}.*?(<span[^>]*>.*?</span>(?:.*?<span[^>]*>.*?</span>)*)'
            m = re.search(pat, html, re.DOTALL | re.IGNORECASE)
            if m:
                spans = re.findall(r'<span[^>]*>([^<]+)</span>', m.group(1))
                section.extend(html_lib.unescape(s.strip()) for s in spans if s.strip())
                break

    return (
        ",".join(top[:8]),
        ",".join(middle[:10]),
        ",".join(base[:8]),
    )


def _extract_meta_from_html(html: str, url: str):
    """Extract name, brand, family from Fragrantica HTML."""
    import html as html_lib

    name  = re.search(r'<h1[^>]*itemprop="name"[^>]*>([^<]+)</h1>', html)
    brand = re.search(r'<span[^>]*itemprop="name"[^>]*>([^<]+)</span>', html)
    fam   = re.search(r'Main Accord.*?<span[^>]*>([^<]+)</span>', html, re.DOTALL)

    name  = html_lib.unescape(name.group(1).strip())  if name  else "Unknown"
    brand = html_lib.unescape(brand.group(1).strip()) if brand else "Unknown"
    fam   = html_lib.unescape(fam.group(1).strip().lower().replace(" ","_")) if fam else "unknown"

    return name, brand, fam


def _compute_vad(top, middle, base, note_emotions):
    notes = [n.strip().lower().replace(" ","_").replace("-","_")
             for n in (top+","+middle+","+base).split(",") if n.strip()]
    V=A=D=count=0.0
    for n in notes:
        if n in note_emotions:
            e = note_emotions[n]
            V+=e["valence"]; A+=e["arousal"]; D+=e["dominance"]; count+=1
    if count==0: return 0.5,0.5,0.5
    return round(V/count,3), round(A/count,3), round(D/count,3)


def scrape_perfume(url: str, session: requests.Session, note_emotions: dict) -> dict | None:
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        html = r.text
        name, brand, fam = _extract_meta_from_html(html, url)
        top, middle, base = _extract_notes_from_html(html)
        if not name or name == "Unknown":
            return None
        V,A,D = _compute_vad(top, middle, base, note_emotions)
        return {"name":name,"brand":brand,"family":fam,
                "top_notes":top,"middle_notes":middle,"base_notes":base,
                "mood_tags":"","valence":V,"arousal":A,"dominance":D,"source":"scraped"}
    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None


def get_designer_perfume_urls(designer_slug: str, session: requests.Session) -> list:
    url = f"https://www.fragrantica.com/designers/{designer_slug}.html"
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        hrefs = re.findall(r'href="(/perfume/[^"]+\.html)"', r.text)
        full = [f"https://www.fragrantica.com{h}" for h in set(hrefs)]
        return full[:50]  # max 50 per designer
    except:
        return []


def load_existing(path: Path) -> set:
    existing = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(row["name"].strip().lower())
    return existing


def run(pages: int = 10, target_url: str = None):
    with open(FAMILIES_F) as f:
        note_emotions = json.load(f)["emotion_profile"]

    existing = load_existing(OUT_CSV)
    new_count = 0

    # Get current max ID
    max_id = 0
    if OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try: max_id = max(max_id, int(row["id"]))
                except: pass

    session = requests.Session()

    write_mode = "a" if OUT_CSV.exists() else "w"
    with open(OUT_CSV, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_mode == "w":
            writer.writeheader()

        if target_url:
            # Single URL mode
            print(f"Scraping: {target_url}")
            entry = scrape_perfume(target_url, session, note_emotions)
            if entry and entry["name"].lower() not in existing:
                max_id += 1
                entry["id"] = max_id
                writer.writerow(entry)
                print(f"  Added: {entry['name']} by {entry['brand']}")
            return

        # Designer sweep mode
        perfume_urls = []
        designers_to_use = DESIGNER_SLUGS[:max(1, pages // 10)]
        for slug in designers_to_use:
            print(f"Fetching designer: {slug}")
            urls = get_designer_perfume_urls(slug, session)
            perfume_urls.extend(urls)
            time.sleep(random.uniform(0.8, 1.5))

        # Deduplicate
        perfume_urls = list(set(perfume_urls))
        random.shuffle(perfume_urls)
        target_count = pages * 20  # ~20 perfumes per page estimate

        print(f"\nScraping up to {len(perfume_urls)} perfume URLs...")
        for i, url in enumerate(perfume_urls[:target_count]):
            entry = scrape_perfume(url, session, note_emotions)
            if entry and entry["name"].lower() not in existing:
                max_id += 1
                entry["id"] = max_id
                writer.writerow(entry)
                existing.add(entry["name"].lower())
                new_count += 1
                if new_count % 20 == 0:
                    print(f"  Progress: {new_count} new perfumes added ({i+1}/{min(len(perfume_urls),target_count)} URLs)")
                f.flush()
            time.sleep(random.uniform(0.8, 1.8))

    print(f"\nDone. Added {new_count} new perfumes to {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=10,
                        help="Approx pages to scrape (10=~200, 50=~1000, 500=~10000 perfumes)")
    parser.add_argument("--url", type=str, default=None,
                        help="Scrape a single Fragrantica URL")
    args = parser.parse_args()
    run(pages=args.pages, target_url=args.url)
