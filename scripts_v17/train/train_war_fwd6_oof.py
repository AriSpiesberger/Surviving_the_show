"""Forward-6-year WAR regressor — OOF evaluation on the v2.0b landmark panel.

Target framing (chosen 2026-06):
  * "Next-6-year WAR": at landmark year S, y = sum of bWAR over seasons
    S+1 .. S+6.  Used only when the window is fully observed (S+6 <= LATEST,
    i.e. S <= 2019) so labels are never right-censored.
  * "Conditional on debut": train + evaluate only on players who reached MLB
    (present in season_war). At deploy, E[fwd6 WAR] = P(debut) * this head,
    mirroring score_war.py's composition with the hazard model.

This reuses, verbatim, the discrete-hazard machinery so the WAR head is
directly comparable to the conditionals:
  * features  : scratch/v20b_oof/panel_cache.npz  (X_lm, pids, S_yrs)
  * OOF folds : scratch/v20b_oof/fold{k}_pids.txt + train{k}_pids.txt
WAR OOF predictions therefore land on the same held-out players as the
discrete OOF, and we join the two on (player_id, snap_year == S).

Usage:
    python -m scripts_v17.train.train_war_fwd6_oof [--k 6] [--horizon 6]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SCRATCH = REPO / "scratch" / "v20b_oof"
PANEL_NPZ = SCRATCH / "panel_cache.npz"
DISCRETE_OOF = REPO / "results" / "training" / "v2.0b_oof_stacked_long.csv"
OUT_CSV = REPO / "evaluation" / "war_fwd6" / "war_fwd6_oof.csv"
LATEST_COMPLETE_YEAR = 2025


def _spearman(a, b):
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _metrics(y, p):
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else float("nan")
    rho = _spearman(y, p)
    return mae, rmse, r, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "prospects_snapshot.db"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    H = args.horizon

    # ---- features / panel ----
    z = np.load(PANEL_NPZ, allow_pickle=True)
    X = z["X_lm"]
    pids = z["pids"].astype(object)
    S = z["S_yrs"].astype(int)
    print(f"panel: X={X.shape}  rows={len(pids):,}")

    # ---- per-season WAR + debut year ----
    conn = sqlite3.connect(args.db)
    sw: dict[str, dict[int, float]] = {}
    for pid, yr, war in conn.execute(
            "SELECT player_id, season_year, war FROM season_war"):
        sw.setdefault(pid, {})[int(yr)] = float(war)
    debut = {pid: dy for pid, dy in conn.execute(
        "SELECT player_id, mlb_debut_year FROM career_outcomes "
        "WHERE mlb_debut_year IS NOT NULL")}
    conn.close()
    war_pids = set(sw)
    print(f"season_war: {len(war_pids):,} debuted+matched players")

    # ---- forward-H label + usable mask (debuted, fully-observed window) ----
    y = np.zeros(len(pids), dtype=np.float64)
    usable = np.zeros(len(pids), dtype=bool)
    for i in range(len(pids)):
        pid = pids[i]
        if pid not in war_pids:
            continue
        s = int(S[i])
        if s + H > LATEST_COMPLETE_YEAR:
            continue
        usable[i] = True
        seasons = sw[pid]
        y[i] = sum(seasons.get(yr, 0.0) for yr in range(s + 1, s + H + 1))
    print(f"usable rows: {int(usable.sum()):,}  "
          f"(y mean={y[usable].mean():.2f} median={np.median(y[usable]):.2f} "
          f"p90={np.percentile(y[usable],90):.2f} max={y[usable].max():.2f})")

    # ---- OOF folds (same player partition as the discrete hazards) ----
    fold_sets = [set(Path(SCRATCH / f"fold{k}_pids.txt").read_text().split())
                 for k in range(args.k)]
    train_sets = [set(Path(SCRATCH / f"train{k}_pids.txt").read_text().split())
                  for k in range(args.k)]

    oof_pred = np.full(len(pids), np.nan)
    for k in range(args.k):
        tr = np.array([usable[i] and (pids[i] in train_sets[k])
                       for i in range(len(pids))])
        te = np.array([usable[i] and (pids[i] in fold_sets[k])
                       for i in range(len(pids))])
        if tr.sum() == 0 or te.sum() == 0:
            continue
        reg = HistGradientBoostingRegressor(
            max_iter=400, max_depth=6, learning_rate=0.04,
            min_samples_leaf=20, l2_regularization=1.0,
            loss="absolute_error", random_state=args.seed + k)
        reg.fit(X[tr], y[tr])
        oof_pred[te] = reg.predict(X[te])
        print(f"  fold {k}: train={int(tr.sum()):,} test={int(te.sum()):,}")

    scored = ~np.isnan(oof_pred)
    yt, yp = y[scored], oof_pred[scored]
    print(f"\nOOF scored rows: {int(scored.sum()):,}")

    # ---- save OOF ----
    out = pd.DataFrame({
        "player_id": pids[scored],
        "snap_year": S[scored],
        "debut_year": [debut.get(p) for p in pids[scored]],
        "y_fwd6_war": np.round(yt, 3),
        "pred_fwd6_war": np.round(yp, 3),
    })
    out["years_to_debut"] = out["debut_year"] - out["snap_year"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # ---- headline + sliced metrics ----
    def report(name, m):
        if m.sum() < 10:
            print(f"  {name:<34} n={int(m.sum()):<6} (too few)")
            return
        mae, rmse, r, rho = _metrics(out["y_fwd6_war"][m].values,
                                     out["pred_fwd6_war"][m].values)
        print(f"  {name:<34} n={int(m.sum()):<6} MAE={mae:5.2f} "
              f"RMSE={rmse:5.2f} r={r:5.3f} spearman={rho:5.3f}")

    print("\n=== forward-6yr WAR | conditional on debut — OOF metrics ===")
    report("ALL usable landmarks", np.ones(len(out), bool))
    ytd = out["years_to_debut"]
    report("  window has MLB yrs (S+6 >= debut)", (out["snap_year"] + H >= out["debut_year"]).values)
    report("  last MiLB yr (S == debut-1)", (ytd == 1).values)
    report("  pre-debut (S < debut)", (ytd > 0).values)
    report("  in-MLB vantage (S >= debut)", (ytd <= 0).values)
    report("  realized value (y > 0)", (out["y_fwd6_war"] > 0).values)
    report("  realized big (y >= 5)", (out["y_fwd6_war"] >= 5).values)

    # ---- comparison vs discrete conditionals ----
    if DISCRETE_OOF.exists():
        print("\n=== vs discrete conditionals (join on player_id, snap_year) ===")
        disc = pd.read_csv(DISCRETE_OOF, usecols=[
            "player_id", "snap_year", "p_ESTABLISHED_MLB", "p_STAR_PLUS_ELITE"])
        m = out.merge(disc, on=["player_id", "snap_year"], how="inner")
        print(f"  joined rows: {len(m):,}")
        if len(m) > 100:
            for col in ("p_ESTABLISHED_MLB", "p_STAR_PLUS_ELITE"):
                rho_pred = _spearman(m["pred_fwd6_war"].values, m[col].values)
                rho_real = _spearman(m["y_fwd6_war"].values, m[col].values)
                print(f"  spearman(pred_fwd6_war, {col:<18}) = {rho_pred:5.3f}   "
                      f"spearman(realized, {col:<18}) = {rho_real:5.3f}")
            # Does continuous WAR add ranking signal beyond P(established)?
            # Among rows the classifier rates similar (mid band), does WAR rank?
            band = (m["p_ESTABLISHED_MLB"] > 0.4) & (m["p_ESTABLISHED_MLB"] < 0.7)
            if band.sum() > 100:
                rho = _spearman(m.loc[band, "pred_fwd6_war"].values,
                                m.loc[band, "y_fwd6_war"].values)
                print(f"  within p_ESTABLISHED in [0.4,0.7] band (n={int(band.sum()):,}): "
                      f"spearman(pred, realized)={rho:.3f}")

    print(f"\nwrote OOF -> {OUT_CSV}")


if __name__ == "__main__":
    main()
