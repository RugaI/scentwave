# -*- coding: utf-8 -*-
"""
ScentWave — Full Gradio App  (v2 — Above & Beyond)

Tabs:
  1. Discover        — song → real match + novel formula + luxury description
  2. Scent Timeline  — song arc over 8 segments
  3. Playlist        — up to 5 songs → named scent collection
  4. Reverse         — perfume name → playlist recommendations
  5. Scent Card      — shareable luxury card visual
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.models.scentwave      import ScentWave
from src.data.features         import extract_features, PRESET_SONGS
from src.data.fragrance_db     import get_db, CANONICAL_NOTES
from src.inference.pipeline    import (
    timeline, playlist, reverse, scent_card, _luxury_description
)

CKPT_PATH = Path(__file__).parent.parent / "checkpoints" / "best.pt"

FAMILY_COLORS = {
    "citrus": "#FFD700", "floral": "#FF69B4", "woody": "#8B4513",
    "oriental": "#9B59B6", "musk": "#BDC3C7", "fresh": "#00CED1",
    "spicy": "#FF4500", "gourmand": "#D2691E", "leather": "#5D4037",
}

GENRE_GROUPS = {
    "Classical":   ["moonlight sonata","clair de lune","four seasons spring","canon in d","symphony no 9 beethoven","nocturne op 9 chopin"],
    "Jazz":        ["take five","so what miles davis","fly me to the moon","autumn leaves","feeling good nina simone"],
    "Pop":         ["blinding lights","shape of you","bad guy","someone like you","rolling in the deep","uptown funk","perfect ed sheeran","levitating dua lipa","watermelon sugar","as it was harry styles"],
    "Rock":        ["bohemian rhapsody","hotel california","smells like teen spirit","stairway to heaven","wish you were here","sweet child o mine","november rain","paint it black"],
    "Hip-Hop/R&B": ["gods plan drake","alright kendrick","bitch better have my money","redbone childish gambino","lose yourself eminem","no role modelz"],
    "Electronic":  ["one more time daft punk","strobe deadmau5","levels avicii","animals martin garrix","midnight city m83","sandstorm darude"],
    "Ambient":     ["lofi chill beats","weightless marconi union","experience ludovico","avril 14th aphex twin"],
    "Soul/Blues":  ["superstition stevie wonder","what a wonderful world","hallelujah jeff buckley","the thrill is gone bb king"],
    "Metal":       ["master of puppets","paranoid black sabbath","through the fire flames"],
    "Country":     ["jolene dolly parton","the gambler kenny rogers","take me home country roads"],
    "Latin/World": ["la bamba","despacito","samba de uma nota so","oye como va"],
}

# ── Load model ───────────────────────────────────────────────────────────────
def _load():
    db    = get_db()
    model = ScentWave()
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        sd   = {k: v for k, v in ckpt["state_dict"].items() if "perfume_embeddings" not in k}
        model.load_state_dict(sd, strict=False)
        print(f"Checkpoint loaded (epoch {ckpt.get('epoch','?')}): {CKPT_PATH}")
    else:
        print("No checkpoint — using untrained model. Run: python train.py")
    model.eval()
    model.init_db(db)
    return model, db

MODEL, DB = _load()

# ── Chart helpers ────────────────────────────────────────────────────────────
BG = "#0d0d0d"
PAPER = "#111111"

def _style(fig, height=300):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PAPER,
        font=dict(color="white", size=12),
        margin=dict(l=10, r=10, t=30, b=30),
        height=height,
    )
    return fig

def emotion_radar(V, A, D):
    cats = ["Valence","Arousal","Dominance","Valence"]
    fig = go.Figure(go.Scatterpolar(
        r=[V,A,D,V], theta=cats, fill="toself",
        fillcolor="rgba(138,43,226,0.25)",
        line=dict(color="#8B00FF", width=2),
        name="",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor="#333"),
                   angularaxis=dict(gridcolor="#333")),
        showlegend=False,
        paper_bgcolor=BG, font=dict(color="white",size=13),
        margin=dict(l=20,r=20,t=20,b=20), height=280,
    )
    return fig

def pyramid_chart(top, middle, base, max_notes=7):
    rows  = ([("TOP",    n, p, "#FFD700") for n,p in top[:max_notes]]
           + [("HEART",  n, p, "#FF69B4") for n,p in middle[:max_notes]]
           + [("BASE",   n, p, "#8B4513") for n,p in base[:max_notes]])
    if not rows: return go.Figure()
    labels = [f"{n}  {p}%" for _,n,p,_ in rows]
    values = [p for _,_,p,_ in rows]
    colors = [c for _,_,_,c in rows]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.4)", width=1)),
        text=[f"{p}%" for p in values], textposition="inside",
        insidetextanchor="middle",
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed", gridcolor="#222"),
        xaxis=dict(gridcolor="#222", range=[0, max(values)*1.25] if values else [0,10]),
        height=max(280, len(rows)*30+60),
    )
    return _style(fig)

def family_donut(profile):
    items = [(k,v) for k,v in profile.items() if v > 1.0]
    if not items: return go.Figure()
    labels = [k.replace("_"," ").title() for k,_ in items]
    values = [v for _,v in items]
    colors = [FAMILY_COLORS.get(k,"#666") for k,_ in items]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textinfo="label+percent",
    ))
    fig.update_layout(showlegend=False, paper_bgcolor=BG,
                      font=dict(color="white",size=11),
                      margin=dict(l=5,r=5,t=5,b=5), height=260)
    return fig

def timeline_chart(segments):
    names = [s["segment"] for s in segments]
    V = [s["emotion"]["valence"]   for s in segments]
    A = [s["emotion"]["arousal"]   for s in segments]
    D = [s["emotion"]["dominance"] for s in segments]

    fig = go.Figure()
    for label, vals, color in [("Valence",V,"#FFD700"),("Arousal",A,"#00CED1"),("Dominance",D,"#FF69B4")]:
        fig.add_trace(go.Scatter(
            x=names, y=vals, mode="lines+markers",
            name=label, line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            fill="tozeroy", fillcolor=color.replace(")",",0.08)").replace("rgb","rgba").replace("#","rgba(").replace(")","") + "0.08)",
        ))
    fig.update_layout(
        xaxis=dict(gridcolor="#222"), yaxis=dict(range=[0,1], gridcolor="#222"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1),
        height=320,
    )
    return _style(fig)

def timeline_notes_chart(segments):
    """Stacked area chart of top-note families across segments."""
    seg_names = [s["segment"] for s in segments]
    families  = list(FAMILY_COLORS.keys())
    # rebuild per-segment family values from notes
    seg_fam   = []
    for s in segments:
        fam_counts = {f: 0.0 for f in families}
        for note, pct in (s["top_notes"] + s["base_notes"]):
            note_key = note.lower().replace(" ","_")
            from src.data.fragrance_db import get_db as _gdb
            db2 = _gdb()
            for fam, fam_notes in db2.families.items():
                if note_key in fam_notes and fam in fam_counts:
                    fam_counts[fam] += pct
        seg_fam.append(fam_counts)

    fig = go.Figure()
    for fam in families:
        vals = [sf[fam] for sf in seg_fam]
        if max(vals) < 0.5: continue
        fig.add_trace(go.Scatter(
            x=seg_names, y=vals, name=fam.title(),
            stackgroup="one", fill="tonexty",
            line=dict(color=FAMILY_COLORS[fam], width=0),
            fillcolor=FAMILY_COLORS[fam] + "99",
        ))
    fig.update_layout(
        xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h"),
        height=280,
    )
    return _style(fig)

def scent_card_fig(card: dict):
    """Renders the shareable scent card as a Plotly figure."""
    fig = go.Figure()
    e   = card["emotion"]

    # Background
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=6,
                  fillcolor="#0a0a0a", line=dict(color="#8B00FF", width=2))

    # Gold rule lines
    for y in [5.2, 0.8]:
        fig.add_shape(type="line", x0=0.4, y0=y, x1=9.6, y1=y,
                      line=dict(color="#FFD700", width=1))

    fig.add_annotation(x=5, y=5.6, text="SCENTWAVE", font=dict(size=14, color="#8B00FF"),
                       showarrow=False, xanchor="center")
    fig.add_annotation(x=5, y=4.9, text=card["formula_name"],
                       font=dict(size=28, color="#FFD700", family="serif"),
                       showarrow=False, xanchor="center")
    fig.add_annotation(x=5, y=4.3, text=f'For: "{card["song_name"]}"',
                       font=dict(size=13, color="#aaa"), showarrow=False, xanchor="center")
    fig.add_annotation(x=5, y=3.75, text=card["tagline"].upper(),
                       font=dict(size=10, color="#888"), showarrow=False, xanchor="center")

    # Note columns
    col_x = [2.0, 5.0, 8.0]
    col_labels = ["TOP NOTES", "BASE NOTES", "CLOSEST MATCH"]
    col_vals = [
        "\n".join(card["top_notes"][:3]),
        "\n".join(card["base_notes"][:2]),
        card["best_match"],
    ]
    for cx, lbl, val in zip(col_x, col_labels, col_vals):
        fig.add_annotation(x=cx, y=3.1, text=lbl,
                           font=dict(size=9, color="#FFD700"), showarrow=False, xanchor="center")
        fig.add_annotation(x=cx, y=2.5, text=val,
                           font=dict(size=11, color="white"), showarrow=False,
                           xanchor="center", align="center")

    # Emotion bar
    for i, (label, val, color) in enumerate([
        ("VALENCE",   e["valence"],   "#FFD700"),
        ("AROUSAL",   e["arousal"],   "#00CED1"),
        ("DOMINANCE", e["dominance"], "#FF69B4"),
    ]):
        bx = 1.2 + i * 2.8
        fig.add_shape(type="rect", x0=bx, y0=1.3, x1=bx+2.2, y1=1.55,
                      fillcolor="#222", line_width=0)
        fig.add_shape(type="rect", x0=bx, y0=1.3, x1=bx+val*2.2, y1=1.55,
                      fillcolor=color, line_width=0)
        fig.add_annotation(x=bx+1.1, y=1.65, text=label,
                           font=dict(size=8, color=color), showarrow=False, xanchor="center")

    fig.add_annotation(x=5, y=0.5,
                       text=f"{card['mood']} · {card['intensity']} · {card['family']}",
                       font=dict(size=10, color="#555"), showarrow=False, xanchor="center")

    fig.update_layout(
        xaxis=dict(visible=False, range=[0,10]),
        yaxis=dict(visible=False, range=[0,6]),
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        margin=dict(l=0,r=0,t=0,b=0), height=360, width=720,
        showlegend=False,
    )
    return fig

# ── Core functions ────────────────────────────────────────────────────────────

def run_discover(song_name, artist, audio_file):
    song_name = (song_name or "").strip()
    artist    = (artist    or "").strip()
    if not song_name and audio_file is None:
        err = "<p style='color:#ff4444'>Please enter a song name or upload an audio file.</p>"
        return err, None, "", None, None, "", None

    features, meta = extract_features(song_name or None, artist, audio_file)
    result  = MODEL.predict(features, top_k=3, note_threshold=0.018)
    card    = scent_card(meta["name"], result)
    e       = result["emotion"]
    gen     = result["generated"]
    ret     = result["retrieved"]

    luxury_desc = _luxury_description(
        meta["name"], e, gen["top"], gen["middle"], gen["base"],
        gen["family_profile"],
    )

    # Song card HTML
    song_html = f"""
    <div style="background:#111;padding:18px;border-radius:14px;border:1px solid #8B00FF">
      <h3 style="color:#FFD700;margin:0 0 4px">{meta['name']}</h3>
      <p style="color:#888;margin:0 0 12px">{meta['artist']}</p>
      <div style="display:flex;gap:24px">
        <span>Valence&nbsp;<b style="color:#FFD700">{e['valence']:.2f}</b></span>
        <span>Arousal&nbsp;<b style="color:#00CED1">{e['arousal']:.2f}</b></span>
        <span>Dominance&nbsp;<b style="color:#FF69B4">{e['dominance']:.2f}</b></span>
      </div>
    </div>"""

    # Retrieved HTML
    medals = ["🥇","🥈","🥉"]
    ret_html = ""
    for p in ret:
        pct = p["match_pct"]
        ret_html += f"""
        <div style="background:#111;padding:14px;border-radius:10px;border:1px solid #222;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span>{medals[p['rank']-1]} <b style="color:#FFD700">{p['name']}</b></span>
            <span style="background:#8B00FF;color:#fff;padding:2px 10px;border-radius:20px;font-size:13px">{pct}%</span>
          </div>
          <p style="color:#777;margin:4px 0;font-size:13px">{p['brand']} · {p['family'].replace('_',' ').title()}</p>
          <div style="background:#222;height:5px;border-radius:4px;margin:8px 0">
            <div style="background:linear-gradient(90deg,#8B00FF,#FF69B4);width:{pct}%;height:100%;border-radius:4px"></div>
          </div>
          <details style="margin-top:6px">
            <summary style="color:#555;cursor:pointer;font-size:12px">Notes ▾</summary>
            <p style="color:#FFD700;font-size:12px;margin:4px 0"><b>Top:</b> {p['top_notes']}</p>
            <p style="color:#FF69B4;font-size:12px;margin:4px 0"><b>Heart:</b> {p['middle_notes']}</p>
            <p style="color:#8B4513;font-size:12px;margin:4px 0"><b>Base:</b> {p['base_notes']}</p>
          </details>
        </div>"""

    # Novel formula HTML
    novel_html = f"""
    <div style="background:#0a1a1a;padding:18px;border-radius:14px;border:1px solid #00CED1;margin-top:12px">
      <div style="display:flex;justify-content:space-between;align-items:start">
        <div>
          <h3 style="color:#00CED1;margin:0 0 4px">✦ {card['formula_name']}</h3>
          <p style="color:#888;margin:0;font-size:13px;font-style:italic">{card['tagline']}</p>
        </div>
        <span style="background:#00CED1;color:#000;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:bold">{card['mood']}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:14px">"""
    for layer, color, label in [("top","#FFD700","TOP"),("middle","#FF69B4","HEART"),("base","#8B4513","BASE")]:
        novel_html += f"""
        <div style="background:#111;padding:10px;border-radius:8px;border-top:3px solid {color}">
          <p style="color:{color};font-size:10px;font-weight:bold;margin:0 0 6px;letter-spacing:2px">{label}</p>"""
        for note, pct in gen[layer][:5]:
            novel_html += f"<p style='margin:2px 0;font-size:12px'>{note} <span style='color:#555'>{pct}%</span></p>"
        novel_html += "</div>"
    novel_html += "</div></div>"

    # Luxury description HTML
    desc_html = "<div style='background:#0d0d1a;padding:18px;border-radius:12px;border-left:3px solid #8B00FF;margin-top:12px'>"
    for line in luxury_desc.split("\n"):
        if not line.strip():
            desc_html += "<br>"
        elif line.startswith("✦") or "No." in line:
            desc_html += f"<p style='color:#00CED1;font-weight:bold;margin:8px 0'>{line}</p>"
        else:
            desc_html += f"<p style='color:#ccc;margin:4px 0;line-height:1.6'>{line.replace('**','<b>').replace('**','</b>')}</p>"
    desc_html += "</div>"

    radar  = emotion_radar(e["valence"], e["arousal"], e["dominance"])
    pyr    = pyramid_chart(gen["top"], gen["middle"], gen["base"])
    donut  = family_donut({k: v for k,v in gen["family_profile"].items() if v > 1})

    return song_html, radar, ret_html, pyr, donut, novel_html, desc_html


def run_timeline(song_name, artist):
    song_name = (song_name or "").strip()
    if not song_name:
        return "<p style='color:#ff4444'>Enter a song name.</p>", None, None
    features, meta = extract_features(song_name, artist or "")
    segs   = timeline(MODEL, features, meta["name"])

    seg_html = f"<h3 style='color:#FFD700'>Scent Arc: {meta['name']}</h3>"
    seg_html += "<div style='display:flex;gap:8px;flex-wrap:wrap;margin:10px 0'>"
    for s in segs:
        fam   = s["family"].replace("_"," ").title()
        color = FAMILY_COLORS.get(s["family"], "#888")
        notes = ", ".join([n for n,_ in s["top_notes"][:2]]) or "—"
        seg_html += f"""
        <div style="background:#111;padding:10px 14px;border-radius:8px;border-top:3px solid {color};min-width:90px">
          <p style="color:{color};font-size:10px;font-weight:bold;margin:0;letter-spacing:1px">{s['segment'].upper()}</p>
          <p style="color:#ccc;font-size:12px;margin:4px 0">{notes}</p>
          <p style="color:#555;font-size:11px;margin:0">{fam}</p>
        </div>"""
    seg_html += "</div>"

    arc   = timeline_chart(segs)
    notes = timeline_notes_chart(segs)
    return seg_html, arc, notes


def run_playlist(*args):
    # args = [song1, artist1, song2, artist2, ..., song5, artist5]
    songs_raw = [(args[i*2], args[i*2+1]) for i in range(5)]
    songs = [(s.strip(), a.strip()) for s,a in songs_raw if s and s.strip()]
    if len(songs) < 2:
        return "<p style='color:#ff4444'>Enter at least 2 song names.</p>", None

    result = playlist(MODEL, songs)
    coll   = result["collection_name"]
    desc   = result["description"]

    html = f"""
    <div style="background:#111;padding:20px;border-radius:14px;border:2px solid #FFD700">
      <h2 style="color:#FFD700;margin:0 0 8px">✦ {coll}</h2>
      <p style="color:#aaa;font-style:italic;margin:0 0 16px">{desc}</p>
      <div style="display:flex;gap:10px;flex-wrap:wrap">"""

    for s in result["songs"]:
        e = s["emotion"]
        html += f"""
        <div style="background:#0d0d0d;padding:12px;border-radius:10px;border:1px solid #333;min-width:150px">
          <p style="color:#FFD700;font-weight:bold;margin:0 0 4px;font-size:14px">{s['song_name']}</p>
          <p style="color:#777;margin:0 0 8px;font-size:12px">{s['artist']}</p>
          <p style="font-size:11px;margin:2px 0">V <b style="color:#FFD700">{e['valence']:.2f}</b>
             · A <b style="color:#00CED1">{e['arousal']:.2f}</b>
             · D <b style="color:#FF69B4">{e['dominance']:.2f}</b></p>
        </div>"""

    html += "</div></div>"

    # Mean emotion radar
    me = result["mean_emotion"]
    radar = emotion_radar(me["valence"], me["arousal"], me["dominance"])
    return html, radar


def run_reverse(perfume_name):
    if not perfume_name or not perfume_name.strip():
        return "<p style='color:#ff4444'>Enter a perfume name.</p>"
    result = reverse(DB, perfume_name.strip())
    if "error" in result:
        return f"<p style='color:#ff4444'>{result['error']}</p>"

    p   = result["perfume"]
    rec = result["recommended_songs"]
    html = f"""
    <div style="background:#111;padding:18px;border-radius:14px;border:1px solid #8B00FF">
      <h3 style="color:#FFD700;margin:0 0 4px">{p['name']}</h3>
      <p style="color:#888;margin:0 0 12px">{p['brand']} · {p['family'].replace('_',' ').title()}</p>
      <p style="color:#aaa;font-style:italic;margin:0 0 16px">{result['description']}</p>
      <h4 style="color:#00CED1;margin:0 0 10px">Recommended Songs</h4>"""
    medals = ["🎵","🎶","🎸","🎹","🎺"]
    for i, s in enumerate(rec):
        html += f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    background:#0d0d0d;padding:10px 14px;border-radius:8px;margin-bottom:8px">
          <span>{medals[i]} {s['song'].title()}</span>
          <span style="background:#8B00FF;color:#fff;padding:2px 10px;border-radius:20px;font-size:12px">{s['match_pct']}%</span>
        </div>"""
    html += "</div>"
    return html


def run_scent_card(song_name, artist):
    song_name = (song_name or "").strip()
    if not song_name:
        return None, "<p style='color:#ff4444'>Enter a song name.</p>"
    features, meta = extract_features(song_name, artist or "")
    result = MODEL.predict(features, top_k=1)
    card   = scent_card(meta["name"], result)
    fig    = scent_card_fig(card)
    info   = f"""
    <div style="text-align:center;padding:10px">
      <h3 style="color:#FFD700">{card['formula_name']}</h3>
      <p style="color:#aaa">{card['tagline']}</p>
      <p style="color:#555;font-size:12px">
        {card['mood']} · {card['intensity']} · {card['family']} · Closest: {card['best_match']}
      </p>
    </div>"""
    return fig, info


# ── UI layout ─────────────────────────────────────────────────────────────────

ALL_PRESETS = sorted(PRESET_SONGS.keys())

with gr.Blocks(title="ScentWave") as demo:
    gr.HTML("""
    <div style="text-align:center;padding:28px 0 8px;background:linear-gradient(180deg,#0d0d0d,transparent)">
      <div style="font-size:11px;color:#8B00FF;letter-spacing:6px;margin-bottom:6px">✦ AI FRAGRANCE INTELLIGENCE ✦</div>
      <h1 style="font-size:3.2em;color:#FFD700;letter-spacing:5px;margin:0;font-weight:300">SCENTWAVE</h1>
      <p style="color:#555;font-size:14px;margin:8px 0 0;letter-spacing:2px">
        ONE SONG · ONE REAL PERFUME · ONE FORMULA NEVER MADE
      </p>
    </div>
    """)

    with gr.Tabs():

        # ── TAB 1: DISCOVER ─────────────────────────────────────────────────
        with gr.TabItem("✦ Discover"):
            with gr.Row():
                with gr.Column(scale=1):
                    song_in   = gr.Textbox(label="Song Name", placeholder="e.g. Clair de Lune")
                    artist_in = gr.Textbox(label="Artist (optional)", placeholder="e.g. Debussy")
                    audio_in  = gr.Audio(label="Or upload audio", type="filepath")
                    run_btn   = gr.Button("Find My Scent", variant="primary", size="lg")

                    gr.HTML("<p style='color:#444;font-size:12px;margin:12px 0 4px;letter-spacing:1px'>QUICK PRESETS BY GENRE</p>")
                    for genre, songs in GENRE_GROUPS.items():
                        with gr.Accordion(genre, open=False):
                            with gr.Row():
                                for s in songs[:4]:
                                    gr.Button(s.title(), size="sm").click(
                                        lambda x=s: (x, ""), outputs=[song_in, artist_in]
                                    )
                            if len(songs) > 4:
                                with gr.Row():
                                    for s in songs[4:8]:
                                        gr.Button(s.title(), size="sm").click(
                                            lambda x=s: (x, ""), outputs=[song_in, artist_in]
                                        )

                with gr.Column(scale=2):
                    song_card_out = gr.HTML()
                    with gr.Row():
                        radar_out = gr.Plot(label="Emotional Fingerprint")
                        donut_out = gr.Plot(label="Scent Family")
                    ret_out   = gr.HTML(label="Closest Real Perfumes")

            gr.HTML("<hr style='border-color:#222;margin:20px 0'>")
            gr.HTML("<div style='text-align:center;color:#00CED1;font-size:16px;letter-spacing:3px'>✦ YOUR SONG'S ORIGINAL FORMULA ✦</div>")
            with gr.Row():
                pyr_out   = gr.Plot(label="Note Pyramid")
            novel_out = gr.HTML()
            desc_out  = gr.HTML()

            run_btn.click(
                fn=run_discover,
                inputs=[song_in, artist_in, audio_in],
                outputs=[song_card_out, radar_out, ret_out, pyr_out, donut_out, novel_out, desc_out],
            )

        # ── TAB 2: SCENT TIMELINE ───────────────────────────────────────────
        with gr.TabItem("⏱ Scent Timeline"):
            gr.HTML("<p style='color:#888;padding:10px 0'>See how your song's scent evolves across 8 segments — intro to fade.</p>")
            with gr.Row():
                tl_song   = gr.Textbox(label="Song Name", placeholder="e.g. Bohemian Rhapsody")
                tl_artist = gr.Textbox(label="Artist (optional)")
                tl_btn    = gr.Button("Analyse Arc", variant="primary")
            tl_segs = gr.HTML()
            tl_arc  = gr.Plot(label="Emotional Arc")
            tl_notes= gr.Plot(label="Scent Family Over Time")

            tl_btn.click(fn=run_timeline, inputs=[tl_song, tl_artist],
                         outputs=[tl_segs, tl_arc, tl_notes])

        # ── TAB 3: PLAYLIST ─────────────────────────────────────────────────
        with gr.TabItem("🎵 Playlist"):
            gr.HTML("<p style='color:#888;padding:10px 0'>Enter up to 5 songs — get a named scent collection that binds them together.</p>")
            pl_inputs = []
            for i in range(5):
                with gr.Row():
                    s = gr.Textbox(label=f"Song {i+1}", placeholder="Song name")
                    a = gr.Textbox(label=f"Artist {i+1}", placeholder="Artist (optional)")
                    pl_inputs += [s, a]
            pl_btn   = gr.Button("Create Collection", variant="primary")
            pl_out   = gr.HTML()
            pl_radar = gr.Plot(label="Collection Mood")
            pl_btn.click(fn=run_playlist, inputs=pl_inputs, outputs=[pl_out, pl_radar])

        # ── TAB 4: REVERSE ──────────────────────────────────────────────────
        with gr.TabItem("🔄 Reverse"):
            gr.HTML("<p style='color:#888;padding:10px 0'>Enter a perfume name — discover the songs that share its soul.</p>")
            with gr.Row():
                rev_in  = gr.Textbox(label="Perfume Name", placeholder="e.g. Dior Sauvage")
                rev_btn = gr.Button("Find Songs", variant="primary")
            rev_out = gr.HTML()
            rev_btn.click(fn=run_reverse, inputs=[rev_in], outputs=[rev_out])

        # ── TAB 5: SCENT CARD ───────────────────────────────────────────────
        with gr.TabItem("🃏 Scent Card"):
            gr.HTML("<p style='color:#888;padding:10px 0'>Generate a luxury scent card for any song — screenshot and share it.</p>")
            with gr.Row():
                sc_song   = gr.Textbox(label="Song Name", placeholder="e.g. Moonlight Sonata")
                sc_artist = gr.Textbox(label="Artist (optional)")
                sc_btn    = gr.Button("Generate Card", variant="primary")
            sc_fig  = gr.Plot(label="")
            sc_info = gr.HTML()
            sc_btn.click(fn=run_scent_card, inputs=[sc_song, sc_artist],
                         outputs=[sc_fig, sc_info])

    gr.HTML("""
    <div style="text-align:center;padding:24px;color:#333;font-size:12px;border-top:1px solid #1a1a1a;margin-top:20px">
      <b style="color:#8B00FF">ScentWave</b> · Music → Scent AI ·
      Train: <code>python train.py</code> ·
      Expand DB: <code>python scripts/build_fragrance_db.py --tier 2 --kaggle data.csv</code>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
