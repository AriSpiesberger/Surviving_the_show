"""Paired held-out comparison of two promoted bundles on identical val rows (2026-09-30).

Each bundle is scored in its own environment (run tag + model DB, so raw features and encoder
tokens come from the DB it was trained on) and saved per row; the comparison then uses ONLY the
rows both share and the labels of the reference run, with a paired player bootstrap.

    # 1. score (once per bundle, in that bundle's environment)
    python -m prospects.evaluation.paired score --xgb runs/current/models/v3.pkl \
        --calibrators runs/current/models/calibrators_v3.pkl --out runs/current/scratch/val_rows_v3.csv
    RUN_TAG=ext3 PROSPECT_MODEL_DB=prospects_ext3.db python -m prospects.evaluation.paired score ...
    # 2. compare (labels from the first file's val long)
    python -m prospects.evaluation.paired compare --ref runs/current/scratch/val_rows_v3.csv \
        --new runs/ext3/scratch/val_rows_v3.csv --labels runs/current/training/oof_val_long.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps

from prospects import config
from prospects.model.joint import EVENTS, prep_base, realized_by_h

HS = (3, 6)


def score(a):
    from prospects.model.joint2 import apply_calibrators_frame, load_calibrators, score_trajectory
    db = str(config.model_db())
    df = prep_base(pd.read_csv(a.val_long or config.run().oof_val_long), db, max_entry=2020)
    df, _ = score_trajectory(a.xgb, df, db)
    if a.calibrators:
        df = apply_calibrators_frame(df, load_calibrators(a.calibrators))
    cols = ["player_id", "snap_year"] + [f"xp_{e}_h{h}" for e in EVENTS for h in HS if f"xp_{e}_h{h}" in df.columns]
    df[cols].to_csv(a.out, index=False)
    print(f"[paired] {len(df):,} rows scored -> {a.out}")


def compare(a):
    from prospects.evaluation.run import EVENT_WEIGHTS
    r = pd.read_csv(a.ref)
    n = pd.read_csv(a.new)
    v = pd.read_csv(a.labels, low_memory=False)
    keep = ["player_id", "snap_year", "years_fwd"] + [c for c in v.columns if c.startswith(("trigger_", "eligible_"))]
    v = v[keep].drop_duplicates(["player_id", "snap_year"])
    m = r.merge(n, on=["player_id", "snap_year"], suffixes=("_ref", "_new")).merge(v, on=["player_id", "snap_year"])
    print(f"[paired] ref {len(r):,} rows, new {len(n):,}, common with labels {len(m):,}")
    rng = np.random.default_rng(0)
    rows = []
    for ev in EVENTS:
        el = (m[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in m else np.ones(len(m), bool)
        for h in HS:
            mm = (m.years_fwd.to_numpy() >= h) & el
            y = np.asarray(realized_by_h(m[mm], ev, h), float)
            if y.sum() < 5:
                continue
            pr, pn = m[f"xp_{ev}_h{h}_ref"].to_numpy()[mm], m[f"xp_{ev}_h{h}_new"].to_numpy()[mm]
            pl = m.player_id.to_numpy()[mm]
            u = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in u}
            d = []
            for _ in range(a.n_boot):
                ix = np.concatenate([pos[q] for q in rng.choice(u, len(u))])
                if y[ix].sum():
                    d.append(aps(y[ix], pn[ix]) - aps(y[ix], pr[ix]))
            rows.append({"event": ev, "h": h, "pos": int(y.sum()), "ap_ref": aps(y, pr), "ap_new": aps(y, pn),
                         "d_ap": aps(y, pn) - aps(y, pr), "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    for h in HS:
        s = df[df.h == h].set_index("event")
        w = pd.Series(EVENT_WEIGHTS).reindex(s.index)
        print(f"weighted AP h{h}: ref {(s.ap_ref * w).sum() / w.sum():.4f}  new {(s.ap_new * w).sum() / w.sum():.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--xgb", required=True)
    s.add_argument("--calibrators", default=None)
    s.add_argument("--val-long", default=None)
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    score(a) if a.cmd == "score" else compare(a)


if __name__ == "__main__":
    main()
