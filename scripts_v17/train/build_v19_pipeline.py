"""v1.19 — train 3 base hazards and score fit/val slices out-of-fold.

Hazards (all binary discrete-time):
  Minors      : first season at AA or AAA            (trigger = year of first AA/AAA stint)
  MLB_service : first MLB appearance                  (trigger = mlb_debut_year)
  All_Star    : first all-star selection              (trigger = year_all_star_once)

The 4 lasso-logit TARGETS (computed into the long, not modeled here):
  TOP_100_Prospect : trigger = year_top_100
  MLB_DEBUT        : trigger = mlb_debut_year
  Established_MLB  : trigger = year_established_mlb
  All_Star_Plus    : trigger = min(year_all_star_once, year_major_award,
                                    year_hof_trajectory)  (whichever fires first)

Split mirrors the v1.17 hazard pipeline:
  cal slice (10%)  = models/event_classifiers_v1.17_lasso_fit_players.txt
  val slice (10%)  = models/event_classifiers_v1.17_lasso_val_players.txt
  train (80%)      = everyone else  — used to fit hazards

The fit + val long files (v1.19h_fit_long.csv / v1.19h_val_long.csv) are
each scored by the 80%-trained hazards (out-of-fold for both slices).
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


HAZARDS = ["Minors", "MLB_service", "All_Star"]
TARGETS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
           "All_Star_Plus"]


def _load_pids(path: str) -> set[str]:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def _reached_aa_year(db: str) -> dict[str, int]:
    """First season per player at level in {AA, AAA}. From season_stats."""
    c = sqlite3.connect(db)
    df = pd.read_sql(
        "SELECT player_id, season_year, level FROM season_stats "
        "WHERE UPPER(level) IN ('AA','AAA')", c)
    c.close()
    df = df.dropna(subset=["season_year"])
    df["season_year"] = df["season_year"].astype(int)
    out: dict[str, int] = {}
    for pid, yr in zip(df["player_id"], df["season_year"]):
        if pid not in out or yr < out[pid]:
            out[pid] = int(yr)
    return out


def _triggers_for_player(p: dict, aa_year: int | None) -> dict[str, int | None]:
    """Trigger years for the 3 hazards + 4 targets keyed by name."""
    def _y(k):
        v = p.get(k)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    debut = _y("mlb_debut_year")
    as1 = _y("year_all_star_once")
    major = _y("year_major_award")
    hof = _y("year_hof_trajectory")
    asp_candidates = [v for v in (as1, major, hof) if v is not None]
    asp = min(asp_candidates) if asp_candidates else None

    return {
        # hazards
        "Minors": aa_year,
        "MLB_service": debut,
        "All_Star": as1,
        # targets
        "TOP_100_PROSPECT": _y("year_top_100"),
        "MLB_DEBUT": debut,
        "ESTABLISHED_MLB": _y("year_established_mlb"),
        "All_Star_Plus": asp,
    }


def _eligible_realized(trigger: int | None, snap_year: int,
                        last_active: int | None,
                        max_obs: int = MAX_OBS_YEAR
                        ) -> tuple[int, int]:
    """Per (player, snap_year):
        eligible = event hasn't already happened by snap_year (and player
                    still has observable future)
        realized = event happens in (snap_year, last observable year]
    """
    if trigger is not None and trigger <= snap_year:
        return 0, 0
    horizon = min(max_obs, last_active) if last_active else max_obs
    if snap_year >= horizon:
        return 0, 0  # no future to predict over
    if trigger is None:
        return 1, 0
    return (1, int(trigger <= horizon))


def _per_row_labels(joined: list[dict], years: np.ndarray,
                     aa_by_pid: dict[str, int],
                     last_active_by_pid: dict[str, int]
                     ) -> dict[str, dict[str, np.ndarray]]:
    """For each event (hazard + target), compute per-row eligible / realized /
    trigger arrays. Hazards use the "realized at year t" form (binary,
    happened exactly this year); targets use the "realized in future" form.
    """
    n = len(years)
    out: dict[str, dict[str, np.ndarray]] = {}
    for ev in HAZARDS:
        out[ev] = {
            "eligible": np.zeros(n, dtype=np.int8),
            "realized": np.zeros(n, dtype=np.int8),
            "trigger": np.full(n, -1, dtype=np.int32),
        }
    for ev in TARGETS:
        out[ev] = {
            "eligible": np.zeros(n, dtype=np.int8),
            "realized": np.zeros(n, dtype=np.int8),
            "trigger": np.full(n, -1, dtype=np.int32),
        }

    for i in range(n):
        p = joined[i]
        yr = int(years[i])
        pid = p["player_id"]
        aa = aa_by_pid.get(pid)
        trigs = _triggers_for_player(p, aa)
        la = last_active_by_pid.get(pid)
        for ev in HAZARDS:
            trig = trigs[ev]
            # Hazard label: eligible if event hasn't yet triggered AND we're
            # not past last-active; realized = 1 if triggered exactly this yr.
            if trig is not None and trig < yr:
                out[ev]["eligible"][i] = 0
            elif la is not None and yr > la and trig is None:
                out[ev]["eligible"][i] = 0
            else:
                out[ev]["eligible"][i] = 1
                if trig is not None and trig == yr:
                    out[ev]["realized"][i] = 1
            if trig is not None:
                out[ev]["trigger"][i] = trig
        for ev in TARGETS:
            elig, real = _eligible_realized(trigs[ev], yr, la)
            out[ev]["eligible"][i] = elig
            out[ev]["realized"][i] = real
            if trigs[ev] is not None:
                out[ev]["trigger"][i] = trigs[ev]
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panels/panel_v1.17.npz")
    ap.add_argument("--joined", default="panels/panel_v1.17.joined.pkl")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--fit-players", required=True,
                    help="10% calibration slice (v1.17 lasso_fit_players)")
    ap.add_argument("--val-players", required=True,
                    help="10% held-out slice (v1.17 lasso_val_players)")
    ap.add_argument("--max-obs", type=int, default=MAX_OBS_YEAR)
    ap.add_argument("--out-model", default="models/hazards_v1.19h.pkl")
    ap.add_argument("--out-fit-long", default="v1.19h_fit_long.csv")
    ap.add_argument("--out-val-long", default="v1.19h_val_long.csv")
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

    # Per-player stats max year (for right-censoring last_active)
    print(f"Loading stats_max_by_pid from {args.db}")
    c = sqlite3.connect(args.db)
    sm = pd.read_sql(
        "SELECT player_id, MAX(season_year) AS y "
        "FROM season_stats GROUP BY player_id", c)
    c.close()
    stats_max = dict(zip(sm["player_id"], sm["y"].astype(int)))

    last_active_by_pid = {p["player_id"]: _last_active(p, stats_max)
                          for p in joined}

    print(f"Loading AA-trigger lookup from {args.db}")
    aa_by_pid = _reached_aa_year(args.db)
    print(f"  {len(aa_by_pid):,} players ever reached AA/AAA")

    cal_pids = _load_pids(args.fit_players)
    val_pids = _load_pids(args.val_players)
    print(f"  cal slice (10%): {len(cal_pids):,} players")
    print(f"  val slice (10%): {len(val_pids):,} players")

    is_train = np.array([p not in cal_pids and p not in val_pids
                         for p in pids], dtype=bool)
    is_fit_score = np.array([p in cal_pids for p in pids], dtype=bool)
    is_val_score = np.array([p in val_pids for p in pids], dtype=bool)
    print(f"  panel rows: train={is_train.sum():,}  "
          f"fit-score={is_fit_score.sum():,}  "
          f"val-score={is_val_score.sum():,}")

    print(f"\nComputing per-row labels for 3 hazards + 4 targets...")
    labels = _per_row_labels(joined, years, aa_by_pid, last_active_by_pid)
    for ev in HAZARDS + TARGETS:
        e = labels[ev]["eligible"].sum()
        r = labels[ev]["realized"].sum()
        print(f"  {ev:<20} eligible={int(e):>8,}  realized={int(r):>7,}")

    # ---- Train 3 hazards on 80% train slice
    print(f"\nTraining 3 hazards on 80% train rows...")
    hazards: dict[str, dict] = {}
    for ev in HAZARDS:
        elig = labels[ev]["eligible"].astype(bool)
        mask = is_train & elig
        Xtr = X[mask]
        ytr = labels[ev]["realized"][mask].astype(int)
        n_pos = int(ytr.sum())
        n = len(ytr)
        if n_pos < MIN_TRAIN_POS or n_pos == n:
            print(f"  [{ev}] SKIP — n_pos={n_pos}/{n:,}")
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=True, n_iter_no_change=20,
            validation_fraction=0.1, random_state=42,
        ).fit(Xtr, ytr)
        hazards[ev] = {"hazard": clf}
        print(f"  [{ev}] n_train={n:,}  pos={n_pos:,} "
              f"({n_pos/n:.2%} base)")

    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    with open(args.out_model, "wb") as fh:
        pickle.dump({"hazards": hazards, "events": list(hazards.keys()),
                     "version": "v1.19", "n_features": N_FEATURES}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out_model}")

    # ---- Score fit + val rows with the 80%-trained hazards
    print(f"\nScoring fit + val rows out-of-fold...")
    p_by_ev: dict[str, np.ndarray] = {}
    for ev in HAZARDS:
        if ev not in hazards:
            p_by_ev[ev] = np.full(len(years), np.nan, dtype=np.float32)
            continue
        proba = hazards[ev]["hazard"].predict_proba(X)[:, 1].astype(np.float32)
        p_by_ev[ev] = proba

    # ---- Assemble long rows for fit + val slices
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

    # bucket (R1/R2-R3/R4-R10/R10+/IFA)
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
    pre_cols["bucket"] = np.array([_bucket(p) for p in joined])

    base = pd.DataFrame(pre_cols)
    for ev in HAZARDS:
        base[f"p_{ev}"] = p_by_ev[ev]
    for ev in HAZARDS + TARGETS:
        base[f"eligible_{ev}"] = labels[ev]["eligible"]
        base[f"realized_{ev}"] = labels[ev]["realized"]
        trig = labels[ev]["trigger"].astype(float)
        trig[trig < 0] = np.nan
        base[f"trigger_{ev}"] = trig

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
