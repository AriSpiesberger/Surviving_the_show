"""Scouting-report text embeddings (2026-09-28).

FanGraphs report text (reference/fangraphs_board/scouting_reports_text.csv: summary + tldr + ovr
summary, one row per player-season board, frozen once a season ends) -> MiniLM sentence
embeddings (mean pooled) -> PCA -> k dims. Joined point-in-time like the scouting grades: a
snapshot year S sees the latest report with season <= S, so the output is expanded to one row per
(player_id, snap_year) and carried forward to MAX_SNAP.

    python tools/build_scout_text_emb.py            # -> runs/experiments/exp_scout_text/scout_text_emb.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FG = ROOT / "reference" / "fangraphs_board"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed(texts, bs=128):
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-snap", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "runs" / "experiments" / "exp_scout_text" / "scout_text_emb.csv"))
    a = ap.parse_args()
    t = pd.read_csv(FG / "scouting_reports_text.csv", low_memory=False)
    t["text"] = (t[["tldr", "summary", "ovr_summary"]].fillna("").astype(str)
                 .agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip())
    t = t[t.text.str.len() > 20].sort_values(["player_id", "season", "source"])
    t = t.drop_duplicates(["player_id", "season"], keep="first").reset_index(drop=True)
    print(f"[text] {len(t):,} reports, {t.player_id.nunique():,} players, seasons {t.season.min()}-{t.season.max()}")
    E = embed(t.text.tolist())
    # PCA fit on reports from seasons <= 2020 only, so the projection never learns from the val-era text mix
    from sklearn.decomposition import PCA
    pca = PCA(n_components=a.k, random_state=0).fit(E[t.season.to_numpy() <= 2020])
    Z = pca.transform(E)
    print(f"[text] PCA {a.k} dims explain {pca.explained_variance_ratio_.sum():.1%}")
    cols = [f"txt_{i}" for i in range(a.k)]
    rep = pd.concat([t[["player_id", "season"]], pd.DataFrame(Z, columns=cols)], axis=1)
    rows = []
    for pid, g in rep.groupby("player_id"):
        g = g.sort_values("season")
        seasons = g.season.to_numpy().astype(int)
        for i, s0 in enumerate(seasons):
            s1 = seasons[i + 1] if i + 1 < len(seasons) else a.max_snap + 1
            for s in range(s0, s1):
                rows.append((pid, s, *g.iloc[i][cols].to_numpy()))
    out = pd.DataFrame(rows, columns=["player_id", "snap_year"] + cols)
    out["txt_present"] = 1.0
    out.to_csv(a.out, index=False)
    print(f"[text] wrote {len(out):,} (player, snap_year) rows -> {a.out}")


if __name__ == "__main__":
    main()
