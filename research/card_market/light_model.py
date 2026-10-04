"""A lightweight, strictly walk-forward stand-in for the production model.

Purpose: honest historical probabilities to test against card prices. The
production OOF predictions on disk predate the 2026-09-17 label-leak fix, and
a full retrain is heavy. This uses the same leak-fixed panel features
(`build_windowed_features`, MiLB-only) with one small gradient-boosting model
per target, trained only on what was known at each season end:

    prediction for season-end Y  <-  landmarks S <= Y-1 (1-year labels)
                                     landmarks S <= Y-2 (2-year labels)

Targets, for players who have not debuted by season S:
    deb1    MLB debut in season S+1
    deb2    MLB debut in S+1 or S+2
    top100  first appearance on a Top-100 list in the S+1 preseason lists

It is NOT the stacked hazard + joint-XGB model; treat results as a floor for
what the production probabilities can do.

    OMP_NUM_THREADS=2 .venv/bin/python research/card_market/light_model.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from prospects.features.partial import partial_for_features
from prospects.model.hazards.landmark import N_FEATURES, _start_year
from prospects.model.hazards.survival import build_windowed_features

HERE = Path(__file__).parent
DB = HERE.parent.parent / "prospects.db"
PANEL = HERE / "raw" / "light_panel.npz"
OUT = HERE / "raw" / "light_preds.csv"
FIRST_S, LAST_S = 2012, 2025
PREDICT_YEARS = (2022, 2023, 2024, 2025)


def build_panel():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    prospects = [dict(r) for r in con.execute(
        """SELECT p.*, o.mlb_debut_year, o.year_established_mlb, o.year_top_100, o.year_top_25,
                  o.year_all_star_once, o.year_all_star_three, o.year_major_award,
                  o.year_hof_trajectory, o.events_json, o.final_mlb_year
           FROM prospects p JOIN career_outcomes o ON o.player_id = p.player_id
           WHERE (p.draft_year IS NOT NULL AND p.draft_year <= 2024)
              OR COALESCE(p.is_international, 0) = 1""")]
    stats_by_pid: dict[str, list[dict]] = {}
    for s in con.execute("SELECT * FROM season_stats"):
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    ranks: dict[str, list] = {}
    for r in con.execute("SELECT player_id, CAST(substr(as_of,1,4) AS INTEGER), overall_rank, source "
                         "FROM rankings_history WHERE overall_rank IS NOT NULL"):
        ranks.setdefault(r[0], []).append((r[1], r[2], r[3]))
    org: dict[str, list] = {}
    for r in con.execute("SELECT player_id, as_of, org_rank FROM rankings_history WHERE org_rank IS NOT NULL"):
        org.setdefault(r[0], []).append((int(str(r[1])[:4]), int(r[2])))

    rows, meta = [], []
    for p in prospects:
        pid = p["player_id"]
        p["_top100_rankings"] = ranks.get(pid, [])
        p["_org_rankings"] = org.get(pid, [])
        stats = stats_by_pid.get(pid, [])
        sy = _start_year(p, stats_by_pid)
        if sy is None:
            continue
        deb, t100 = p.get("mlb_debut_year"), p.get("year_top_100")
        for S in range(max(sy + 1, FIRST_S), LAST_S + 1):
            if deb is not None and deb <= S:
                break  # only not-yet-debuted landmarks
            if S - sy > 9:
                break
            rows.append((p, stats, S))
            meta.append((pid, S, sy, deb if deb is not None else np.nan, t100 if t100 is not None else np.nan))
    X = np.empty((len(rows), N_FEATURES), dtype=np.float32)
    for i, (p, stats, S) in enumerate(rows):
        X[i] = build_windowed_features(p, partial_for_features(stats, S, p["player_id"], None), S, milb_only=True)
    m = pd.DataFrame(meta, columns=["player_id", "S", "start", "debut", "top100"])
    np.savez_compressed(PANEL, X=X, **{c: m[c].to_numpy() for c in m.columns})
    return X, m


def load_panel():
    if not PANEL.exists():
        return build_panel()
    z = np.load(PANEL, allow_pickle=True)
    return z["X"], pd.DataFrame({c: z[c] for c in ("player_id", "S", "start", "debut", "top100")})


def fit_predict(X, y, train, test):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    clf.fit(X[train], y[train])
    return clf.predict_proba(X[test])[:, 1]


def main():
    X, m = load_panel()
    print(f"panel {X.shape}; landmarks {int(m.S.min())}-{int(m.S.max())}; players {m.player_id.nunique():,}")
    S = m.S.to_numpy()
    y = {"deb1": (m.debut == m.S + 1).to_numpy(),
         "deb2": ((m.debut == m.S + 1) | (m.debut == m.S + 2)).to_numpy(),
         "top100": (m.top100 == m.S + 1).to_numpy()}
    elig_top = (m.top100.isna() | (m.top100 > m.S)).to_numpy()  # not already a Top-100 name
    lag = {"deb1": 1, "deb2": 2, "top100": 1}
    out = []
    for Y in PREDICT_YEARS:
        test = S == Y
        d = m.loc[test, ["player_id", "S", "start", "debut", "top100"]].copy()
        for k in y:
            train = S <= Y - lag[k]
            if k == "top100":
                train = train & elig_top
            d["p_" + k] = fit_predict(X, y[k], train, test)
            d["y_" + k] = y[k][test]
        d["elig_top100"] = elig_top[test]
        out.append(d)
        msg = []
        for k in y:
            yy = d["y_" + k].to_numpy()
            ok = np.ones(len(d), bool) if k != "top100" else d.elig_top100.to_numpy()
            if Y + lag[k] <= 2026 and yy[ok].sum() > 0:
                msg.append(f"{k}: base {yy[ok].mean():.3f} AP {average_precision_score(yy[ok], d['p_' + k][ok]):.3f} "
                           f"AUC {roc_auc_score(yy[ok], d['p_' + k][ok]):.3f}")
        print(Y, f"n={len(d):,} |", " | ".join(msg))
    pd.concat(out).to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
