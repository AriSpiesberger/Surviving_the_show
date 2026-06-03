"""v2.1 — trajectory model.

Instead of binary "reach event X" hazards, we predict the player's level
class at each forward horizon. The lasso then sees the trajectory.

Per (player, year=t) panel row:
  Target at horizon h: level class at year t+h
    Below_AA  : highest level that year in {RK, A-, A, A+} OR didn't play
                (also: if not yet reached AA/AAA/MLB/ALL_STAR by then)
    AA        : highest level that year was AA (and not yet ALL_STAR)
    AAA       : highest level that year was AAA (and not yet ALL_STAR)
    MLB       : highest level that year was MLB (and not yet ALL_STAR)
    ALL_STAR  : sticky — player has been an all-star by year t+h

One HistGradientBoostingClassifier per horizon h ∈ {1..H}.
Trajectory = 5 classes × H horizons probability matrix per snapshot.

Lasso targets (computed into the long, not modeled here):
  TOP_100_PROSPECT, MLB_DEBUT, ESTABLISHED_MLB, All_Star_Plus
"""
from __future__ import annotations

import argparse
import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from prospects.features.scouting import N_FEATURES

MAX_OBS_YEAR = 2025
MIN_TRAIN_POS = 30

LEVELS = ["Below_AA", "AA", "AAA", "MLB", "ALL_STAR"]
HORIZONS = [1, 2, 3, 4, 5]

# Each player-year now gets a 5-d binary indicator vector — multi-label,
# not multi-class. So a player who was promoted AA→AAA→MLB in one year has
# played_AA = played_AAA = played_MLB = 1 simultaneously.
LEVEL_TO_INDICATOR = {
    "RK": "Below_AA", "A-": "Below_AA", "A": "Below_AA", "A+": "Below_AA",
    "AA": "AA", "AAA": "AAA", "MLB": "MLB",
}

TARGETS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
           "All_Star_Plus"]


def _load_pids(path: str) -> set[str]:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def _levels_played_by_year(db: str) -> dict[tuple[str, int], set[str]]:
    """Returns {(player_id, season_year): {indicator_name, ...}} — the set
    of LEVELS the player played at in that year (multi-label)."""
    c = sqlite3.connect(db)
    df = pd.read_sql(
        "SELECT player_id, season_year, level FROM season_stats", c)
    c.close()
    df = df.dropna(subset=["season_year"])
    df["season_year"] = df["season_year"].astype(int)
    df["indicator"] = (df["level"].astype(str).str.upper()
                       .map(LEVEL_TO_INDICATOR))
    df = df.dropna(subset=["indicator"])
    out: dict[tuple[str, int], set[str]] = {}
    for pid, yr, ind in zip(df["player_id"], df["season_year"],
                              df["indicator"]):
        key = (pid, int(yr))
        s = out.get(key)
        if s is None:
            out[key] = {ind}
        else:
            s.add(ind)
    return out


def _build_targets_per_row(
        joined: list[dict], years: np.ndarray,
        levels_by_py: dict[tuple[str, int], set[str]],
        max_obs: int = MAX_OBS_YEAR,
    ) -> dict[tuple[int, str], np.ndarray]:
    """For each (horizon h, level_name), return a binary array (n,) with:
       1 = player played at that level in year t+h (or, for ALL_STAR,
           had ever been an all-star by year t+h)
       0 = did not
      -1 = unknown (year t+h > max_obs), excluded from training/eval
    """
    n = len(years)
    out: dict[tuple[int, str], np.ndarray] = {}
    for h in HORIZONS:
        for lv in LEVELS:
            out[(h, lv)] = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        yr0 = int(years[i])
        p = joined[i]
        pid = p["player_id"]
        try:
            as_y = (int(p["year_all_star_once"])
                    if p.get("year_all_star_once") is not None else None)
        except (TypeError, ValueError):
            as_y = None
        for h in HORIZONS:
            t = yr0 + h
            if t > max_obs:
                continue
            played = levels_by_py.get((pid, t), set())
            for lv in ("Below_AA", "AA", "AAA", "MLB"):
                out[(h, lv)][i] = int(lv in played)
            out[(h, "ALL_STAR")][i] = int(
                as_y is not None and as_y <= t)
    return out


def _eligible_realized(trigger: int | None, snap_year: int,
                        last_active: int | None,
                        max_obs: int = MAX_OBS_YEAR
                        ) -> tuple[int, int]:
    if trigger is not None and trigger <= snap_year:
        return 0, 0
    horizon = min(max_obs, last_active) if last_active else max_obs
    if snap_year >= horizon:
        return 0, 0
    if trigger is None:
        return 1, 0
    return (1, int(trigger <= horizon))


def _last_active(p: dict, stats_max_by_pid: dict[str, int]) -> int | None:
    fy = p.get("final_mlb_year")
    sm = stats_max_by_pid.get(p["player_id"])
    if fy is None and sm is None:
        return None
    if fy is None:
        return int(sm)
    if sm is None:
        return int(fy)
    return int(max(fy, sm))


def _bucket(p):
    if int(p.get("is_international") or 0) == 1:
        return "IFA"
    dr = p.get("draft_round")
    if dr is None:
        return "IFA"
    try:
        dr = int(dr)
    except Exception:
        return "IFA"
    if dr == 1:
        return "R1"
    if dr in (2, 3):
        return "R2-R3"
    if 4 <= dr <= 10:
        return "R4-R10"
    return "R10+"


def _target_triggers(p: dict) -> dict[str, int | None]:
    def _y(k):
        v = p.get(k)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    as1 = _y("year_all_star_once")
    major = _y("year_major_award")
    hof = _y("year_hof_trajectory")
    asp = [v for v in (as1, major, hof) if v is not None]
    return {
        "TOP_100_PROSPECT": _y("year_top_100"),
        "MLB_DEBUT": _y("mlb_debut_year"),
        "ESTABLISHED_MLB": _y("year_established_mlb"),
        "All_Star_Plus": min(asp) if asp else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panels/panel_v1.17.npz")
    ap.add_argument("--joined", default="panels/panel_v1.17.joined.pkl")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--fit-players", required=True)
    ap.add_argument("--val-players", required=True)
    ap.add_argument("--max-obs", type=int, default=MAX_OBS_YEAR)
    ap.add_argument("--out-model", default="models/level_traj_v2.1h.pkl")
    ap.add_argument("--out-fit-long", default="v2.1h_fit_long.csv")
    ap.add_argument("--out-val-long", default="v2.1h_val_long.csv")
    args = ap.parse_args()

    print(f"Loading panel {args.panel}")
    with np.load(args.panel, allow_pickle=True) as d:
        X = d["X"].astype(np.float32, copy=False)
        pids = np.asarray(d["pids"])
        years = np.asarray(d["years"], dtype=int)
    assert X.shape[1] == N_FEATURES
    print(f"  {X.shape[0]:,} rows, {X.shape[1]} features, "
          f"{len(set(pids.tolist())):,} players")

    print(f"Loading joined {args.joined}")
    with open(args.joined, "rb") as fh:
        joined = pickle.load(fh)
    assert len(joined) == X.shape[0]

    print(f"Building per-(pid,year) levels-played set...")
    levels_by_py = _levels_played_by_year(args.db)
    print(f"  {len(levels_by_py):,} (pid, year) entries with recognized level")

    print(f"Loading stats_max_by_pid from {args.db}")
    c = sqlite3.connect(args.db)
    sm = pd.read_sql(
        "SELECT player_id, MAX(season_year) AS y "
        "FROM season_stats GROUP BY player_id", c)
    c.close()
    stats_max = dict(zip(sm["player_id"], sm["y"].astype(int)))
    last_active_by_pid = {p["player_id"]: _last_active(p, stats_max)
                          for p in joined}

    cal_pids = _load_pids(args.fit_players)
    val_pids = _load_pids(args.val_players)
    is_train = np.array([p not in cal_pids and p not in val_pids
                         for p in pids], dtype=bool)
    is_fit_score = np.array([p in cal_pids for p in pids], dtype=bool)
    is_val_score = np.array([p in val_pids for p in pids], dtype=bool)
    print(f"  cal: {len(cal_pids):,}  val: {len(val_pids):,}")
    print(f"  panel rows: train={is_train.sum():,}  "
          f"fit-score={is_fit_score.sum():,}  "
          f"val-score={is_val_score.sum():,}")

    print(f"\nBuilding per-(horizon, level) binary targets...")
    y_by_hl = _build_targets_per_row(joined, years, levels_by_py,
                                       max_obs=args.max_obs)
    for h in HORIZONS:
        for lv in LEVELS:
            y = y_by_hl[(h, lv)]
            valid = y >= 0
            pos = int((y == 1).sum())
            print(f"  h={h} {lv:<10}  n_valid={int(valid.sum()):>7,}  "
                  f"pos={pos:>6,} ({pos/max(int(valid.sum()),1):.2%})")

    print(f"\nTraining 25 binary classifiers (5 horizons × 5 levels) "
          f"on 80% train slice...")
    classifiers: dict[tuple[int, str], HistGradientBoostingClassifier] = {}
    for h in HORIZONS:
        for lv in LEVELS:
            y = y_by_hl[(h, lv)]
            valid = y >= 0
            mask = is_train & valid
            Xtr = X[mask]
            ytr = y[mask].astype(int)
            n_pos = int(ytr.sum())
            if n_pos < MIN_TRAIN_POS or n_pos == len(ytr):
                print(f"  h={h} {lv:<10}  SKIP (pos={n_pos}/{len(ytr):,})")
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=30, l2_regularization=1.0,
                early_stopping=True, n_iter_no_change=20,
                validation_fraction=0.1, random_state=42,
            ).fit(Xtr, ytr)
            classifiers[(h, lv)] = clf
            print(f"  h={h} {lv:<10}  n_train={len(ytr):,}  "
                  f"pos={n_pos:,} ({n_pos/len(ytr):.2%})")

    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    with open(args.out_model, "wb") as fh:
        pickle.dump({
            "classifiers": classifiers,
            "horizons": HORIZONS,
            "levels": LEVELS,
            "version": "v2.1",
            "n_features": N_FEATURES,
            "kind": "multi_label_level_trajectory",
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out_model}")

    print(f"\nScoring panel for fit + val slices (out-of-fold)...")
    score_mask = is_fit_score | is_val_score
    Xs = X[score_mask]
    proba_cols: dict[str, np.ndarray] = {}
    for (h, lv), clf in classifiers.items():
        # Binary classifier: predict_proba[:, 1] = P(played at this level)
        proba = clf.predict_proba(Xs)[:, 1]
        col_name = f"p_h{h}_{lv}"
        full = np.full(len(years), np.nan, dtype=np.float32)
        full[score_mask] = proba
        proba_cols[col_name] = full

    print(f"Assembling long CSVs...")
    pre_cols = {
        "player_id": pids,
        "name": np.array([p.get("name") for p in joined]),
        "draft_year": np.array([p.get("draft_year") for p in joined]),
        "draft_round": np.array([p.get("draft_round") for p in joined]),
        "is_international": np.array(
            [int(p.get("is_international") or 0) for p in joined]),
        "snap_year": years,
        "mlb_debut_year": np.array([p.get("mlb_debut_year") for p in joined]),
    }
    snap_offset = np.array(
        [int(years[i]) - int(joined[i].get("draft_year") or
                              joined[i].get("international_signing_year") or
                              years[i])
         for i in range(len(years))], dtype=int)
    pre_cols["snap_offset"] = snap_offset
    pre_cols["entry_year"] = np.array(
        [int(joined[i].get("draft_year") or
              joined[i].get("international_signing_year") or
              years[i])
         for i in range(len(years))], dtype=int)
    pre_cols["bucket"] = np.array([_bucket(p) for p in joined])

    base = pd.DataFrame(pre_cols)
    for col, arr in proba_cols.items():
        base[col] = arr
    # Target labels (forward-window) for the 4 lasso outcomes
    for i in range(len(years)):
        pass  # vectorized below

    # Vectorized target labels
    print(f"  computing target labels (eligible/realized/trigger)...")
    n = len(years)
    for ev in TARGETS:
        elig = np.zeros(n, dtype=np.int8)
        real = np.zeros(n, dtype=np.int8)
        trig = np.full(n, -1.0, dtype=float)
        for i in range(n):
            tt = _target_triggers(joined[i])
            t = tt[ev]
            yr = int(years[i])
            la = last_active_by_pid.get(joined[i]["player_id"])
            e, r = _eligible_realized(t, yr, la, max_obs=args.max_obs)
            elig[i] = e
            real[i] = r
            if t is not None:
                trig[i] = t
        base[f"eligible_{ev}"] = elig
        base[f"realized_{ev}"] = real
        base[f"trigger_{ev}"] = np.where(trig < 0, np.nan, trig)

    fit_long = base[is_fit_score].copy()
    val_long = base[is_val_score].copy()
    fit_long.to_csv(args.out_fit_long, index=False)
    val_long.to_csv(args.out_val_long, index=False)
    print(f"  wrote {args.out_fit_long}: {len(fit_long):,} rows, "
          f"{fit_long.player_id.nunique():,} players")
    print(f"  wrote {args.out_val_long}: {len(val_long):,} rows, "
          f"{val_long.player_id.nunique():,} players")


if __name__ == "__main__":
    main()
