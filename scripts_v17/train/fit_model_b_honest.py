"""Model B (honest): P(cup/utility/regular/breakout | debut, raw features X).

Parallel to the hazard pipeline:
  - 80% train  = panel players NOT in lasso_fit_players + lasso_val_players
  - 10% calibrate = lasso_fit_players
  - 10% validate  = lasso_val_players  (touched only by validate_model_b.py)

Standalone conditional model — no hazard predictions consumed as features.
Per debutee, take ONE row at the earliest pre-debut snap; label from post-
debut MLB stats. Multinomial LogisticRegression on raw 238-d windowed
scouting features.

Usage:
    python -m scripts_v17.train.fit_model_b_honest \\
        --panel panel_v1.17.npz \\
        --fit-players models/event_classifiers_v1.17_lasso_fit_players.txt \\
        --val-players models/event_classifiers_v1.17_lasso_val_players.txt \\
        --out models/model_b_outcomes_v1.17h.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sqlite3
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from prospects.features.scouting import FEATURE_NAMES, N_FEATURES
from prospects.storage import ProspectDB

# 3-class scheme — debut-year stats + post-debut demotion only:
#   breakout: debut-year MLB stats hit a quality threshold
#             — ERA < 3.80 (with IP >= 20)  OR  AVG > 0.275 (with PA >= 50).
#   cup:      not breakout AND had a MiLB appearance in any season AFTER
#             debut_year (got sent back down).
#   debut:    not breakout AND no MiLB appearance after debut_year.
#             (Clean middling debut — didn't break out, didn't fail back.)
CLASSES = ["cup", "debut", "breakout"]
BREAKOUT_ERA_MAX = 3.80
BREAKOUT_AVG_MIN = 0.275
MIN_IP_FOR_ERA = 20.0
MIN_PA_FOR_AVG = 50
DEBUT_MIN_DEFAULT, DEBUT_MAX_DEFAULT = 2010, 2024


def _load_pids(path: str) -> set[str]:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def _load_all_stats(db: str) -> pd.DataFrame:
    """All season_stats rows needed for labeling:
       - level (to detect MiLB demotion post-debut)
       - pa, avg, ip, era (debut-year quality stats)
    """
    c = sqlite3.connect(db)
    df = pd.read_sql(
        "SELECT player_id, season_year, level, pa, avg, ip, era "
        "FROM season_stats", c)
    c.close()
    df["level_upper"] = df["level"].fillna("").str.upper()
    df["is_milb"] = (df["level_upper"] != "MLB") & (df["level_upper"] != "")
    df["pa"] = df["pa"].fillna(0.0)
    df["ip"] = df["ip"].fillna(0.0)
    return df


# Kept for backward compatibility with existing callers; unused by the new
# label_player.
def _load_mlb_stats(db: str) -> pd.DataFrame:
    df = _load_all_stats(db)
    return df[df.level_upper == "MLB"][["player_id", "season_year"]].copy()


def label_player(rows: pd.DataFrame, debut_year: int) -> str:
    """3-class label — debut-year stats + post-debut demotion only.

    breakout: debut-year MLB stats hit a quality threshold
              — ERA < 3.80 (IP >= 20)  OR  AVG > 0.275 (PA >= 50).
    cup:      not breakout AND any MiLB row in season > debut_year.
    debut:    not breakout AND no MiLB row in season > debut_year.

    `rows` must contain ALL season_stats for this player with columns:
    season_year, level_upper, is_milb, pa, avg, ip, era.
    """
    # Debut-year MLB rows (may be multiple if multi-stint or split-org)
    debut_mlb = rows[(rows.season_year == debut_year) &
                     (rows.level_upper == "MLB")]
    if not debut_mlb.empty:
        pa = float(debut_mlb["pa"].sum())
        ip = float(debut_mlb["ip"].sum())
        # PA-weighted AVG across debut-year MLB rows
        if pa >= MIN_PA_FOR_AVG:
            valid = debut_mlb[debut_mlb["avg"].notna() & (debut_mlb["pa"] > 0)]
            if not valid.empty:
                w_avg = float((valid["avg"] * valid["pa"]).sum() /
                              valid["pa"].sum())
                if w_avg > BREAKOUT_AVG_MIN:
                    return "breakout"
        # IP-weighted ERA across debut-year MLB rows
        if ip >= MIN_IP_FOR_ERA:
            valid = debut_mlb[debut_mlb["era"].notna() & (debut_mlb["ip"] > 0)]
            if not valid.empty:
                w_era = float((valid["era"] * valid["ip"]).sum() /
                              valid["ip"].sum())
                if w_era < BREAKOUT_ERA_MAX:
                    return "breakout"
    # Demotion check: any MiLB row strictly after debut_year
    post = rows[rows.season_year > debut_year]
    if not post.empty and post["is_milb"].any():
        return "cup"
    return "debut"


def _earliest_predebut_idx(pids: np.ndarray, years: np.ndarray,
                            debut_by_pid: dict[str, int]) -> dict[str, int]:
    """Earliest pre-debut row per debutee. Kept for any caller that wants a
    single representative snapshot."""
    best: dict[str, tuple[int, int]] = {}
    for i, (pid, yr) in enumerate(zip(pids, years)):
        d = debut_by_pid.get(pid)
        if d is None or yr >= d:
            continue
        cur = best.get(pid)
        if cur is None or yr < cur[0]:
            best[pid] = (int(yr), i)
    return {pid: idx for pid, (_, idx) in best.items()}


def _latest_predebut_idx(pids: np.ndarray, years: np.ndarray,
                          debut_by_pid: dict[str, int]) -> dict[str, int]:
    """Latest pre-debut row per debutee (closest to debut). Used as the
    single per-player representative for validation."""
    best: dict[str, tuple[int, int]] = {}
    for i, (pid, yr) in enumerate(zip(pids, years)):
        d = debut_by_pid.get(pid)
        if d is None or yr >= d:
            continue
        cur = best.get(pid)
        if cur is None or yr > cur[0]:
            best[pid] = (int(yr), i)
    return {pid: idx for pid, (_, idx) in best.items()}


def _all_predebut_idx(pids: np.ndarray, years: np.ndarray,
                       debut_by_pid: dict[str, int]) -> dict[str, list[int]]:
    """ALL pre-debut panel rows per debutee — each training row labeled with
    the same post-debut outcome. Mirrors how the hazard model uses every
    (player, year) row in the panel.

    Returns {player_id: [row_idx, ...]} sorted by year ascending.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for i, (pid, yr) in enumerate(zip(pids, years)):
        d = debut_by_pid.get(pid)
        if d is None or yr >= d:
            continue
        out.setdefault(pid, []).append((int(yr), i))
    return {pid: [i for _, i in sorted(v)] for pid, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel_v1.17.npz")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--fit-players", required=True,
                    help="10% calibration slice (hazard lasso_fit_players)")
    ap.add_argument("--val-players", required=True,
                    help="10% validation slice (hazard lasso_val_players) — "
                         "held out of both train and calibrate")
    ap.add_argument("--debut-min", type=int, default=DEBUT_MIN_DEFAULT)
    ap.add_argument("--debut-max", type=int, default=DEBUT_MAX_DEFAULT)
    ap.add_argument("--C", type=float, default=1.0,
                    help="LogisticRegression inverse-reg strength")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading panel {args.panel}")
    with np.load(args.panel, allow_pickle=True) as d:
        X_full = d["X"].astype(np.float32, copy=False)
        pids = np.asarray(d["pids"])
        years = np.asarray(d["years"], dtype=int)
    assert X_full.shape[1] == N_FEATURES, (X_full.shape, N_FEATURES)
    print(f"  panel: {X_full.shape[0]:,} rows, "
          f"{len(set(pids.tolist())):,} players, {X_full.shape[1]} features")

    cal_pids = _load_pids(args.fit_players)
    val_pids = _load_pids(args.val_players)
    print(f"  cal slice (10%): {len(cal_pids):,} players")
    print(f"  val slice (10%): {len(val_pids):,} players  "
          f"(HELD OUT — only validate_model_b touches)")

    print(f"\nLoading debut years from {args.db}")
    db = ProspectDB(args.db)
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT player_id, mlb_debut_year FROM career_outcomes "
            "WHERE mlb_debut_year IS NOT NULL").fetchall()
    debut_by_pid = {r["player_id"]: int(r["mlb_debut_year"]) for r in rows
                    if args.debut_min <= int(r["mlb_debut_year"]) <= args.debut_max}
    print(f"  debutees {args.debut_min}-{args.debut_max}: "
          f"{len(debut_by_pid):,}")

    print(f"\nCollecting ALL pre-debut panel rows per debutee...")
    all_pre = _all_predebut_idx(pids, years, debut_by_pid)
    n_snaps = sum(len(v) for v in all_pre.values())
    print(f"  matched {len(all_pre):,} debutees to {n_snaps:,} pre-debut rows "
          f"(avg {n_snaps/max(len(all_pre),1):.1f} snaps/player)")

    print(f"\nBuilding labels from full season stats (MLB + MiLB)...")
    stats = _load_all_stats(args.db)
    print(f"  {len(stats):,} rows ({stats.is_milb.sum():,} MiLB, "
          f"{(~stats.is_milb & (stats.level_upper == 'MLB')).sum():,} MLB), "
          f"{stats.player_id.nunique():,} players")
    stats_by_pid = {pid: g for pid, g in stats.groupby("player_id")}

    # Per-player labels — one label per debutee, replicated across snaps.
    pid_to_label: dict[str, int] = {}
    skipped_no_post = 0
    for pid in all_pre:
        debut = debut_by_pid[pid]
        post = stats_by_pid.get(pid)
        if post is None:
            skipped_no_post += 1
            label = "debut"
        else:
            label = label_player(post, debut)
        pid_to_label[pid] = CLASSES.index(label)
    if skipped_no_post:
        print(f"  warn: {skipped_no_post:,} debutees had no stat rows "
              f"(labeled 'debut')")

    train_rows, cal_rows, val_rows = [], [], []
    train_y, cal_y, val_y = [], [], []
    train_pidlist, cal_pidlist, val_pidlist = [], [], []
    for pid, idx_list in all_pre.items():
        y_k = pid_to_label[pid]
        if pid in val_pids:
            for idx in idx_list:
                val_rows.append(idx); val_y.append(y_k); val_pidlist.append(pid)
        elif pid in cal_pids:
            for idx in idx_list:
                cal_rows.append(idx); cal_y.append(y_k); cal_pidlist.append(pid)
        else:
            for idx in idx_list:
                train_rows.append(idx); train_y.append(y_k)
                train_pidlist.append(pid)

    X_tr = X_full[np.asarray(train_rows)]
    X_ca = X_full[np.asarray(cal_rows)]
    X_va = X_full[np.asarray(val_rows)]
    y_tr = np.asarray(train_y, dtype=int)
    y_ca = np.asarray(cal_y, dtype=int)
    y_va = np.asarray(val_y, dtype=int)
    n_tr_pid = len(set(train_pidlist))
    n_ca_pid = len(set(cal_pidlist))
    n_va_pid = len(set(val_pidlist))
    print(f"\nSlice sizes (snapshots / unique debutees):")
    print(f"  train: {len(y_tr):,} snaps / {n_tr_pid:,} players")
    print(f"  cal:   {len(y_ca):,} snaps / {n_ca_pid:,} players")
    print(f"  val:   {len(y_va):,} snaps / {n_va_pid:,} players")

    print(f"\nClass distribution (train snapshots):")
    for k, name in enumerate(CLASSES):
        n = int((y_tr == k).sum())
        print(f"  {name:<10s} {n:>5d}  {n/len(y_tr):.1%}")

    print(f"\nNaN cells in train X: {int(np.isnan(X_tr).sum()):,} "
          f"({np.isnan(X_tr).mean():.2%}) — HGB handles natively, no impute")

    print(f"\nFitting HistGradientBoostingClassifier on train snapshots...")
    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.0,
        random_state=42,
        early_stopping=True, n_iter_no_change=20,
        validation_fraction=0.1,
    ).fit(X_tr, y_tr)

    p_tr = clf.predict_proba(X_tr)
    ll_tr = -np.log(p_tr[np.arange(len(y_tr)), y_tr] + 1e-9).mean()
    base = np.array([(y_tr == k).mean() for k in range(len(CLASSES))])
    base_ll = -np.log(base[y_tr] + 1e-9).mean()
    print(f"  in-train log-loss: {ll_tr:.4f}   baseline (priors): {base_ll:.4f}")

    # Per-class Platt calibration on cal-slice snapshots (player-grouped
    # already since cal pids are disjoint from train).
    print(f"\nFitting per-class Platt calibrators on cal slice "
          f"({len(y_ca):,} snaps / {n_ca_pid:,} players)...")
    cal_calibrators: list[LogisticRegression | None] = [None] * len(CLASSES)
    if len(y_ca) >= 50:
        p_ca = clf.predict_proba(X_ca)
        from sklearn.metrics import brier_score_loss
        for k, name in enumerate(CLASSES):
            y_k = (y_ca == k).astype(int)
            if y_k.sum() == 0 or y_k.sum() == len(y_k):
                print(f"  {name}: cal slice all-{int(y_k.mean())} — skipped")
                continue
            lr_k = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
            lr_k.fit(p_ca[:, k:k+1], y_k)
            cal_calibrators[k] = lr_k
            p_raw = p_ca[:, k]
            p_cal = lr_k.predict_proba(p_ca[:, k:k+1])[:, 1]
            b_raw = brier_score_loss(y_k, p_raw)
            b_cal = brier_score_loss(y_k, p_cal)
            print(f"  {name:<10s} brier raw={b_raw:.4f}  cal={b_cal:.4f}  "
                  f"(base rate {y_k.mean():.1%})")
    else:
        print(f"  cal slice too small ({len(y_ca)}), skipping calibration")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    artifact = {
        "model": clf,
        "scaler": None,  # HGB doesn't need scaling
        "feature_names": list(FEATURE_NAMES),
        "classes": CLASSES,
        "cal_calibrators": cal_calibrators,
        "train_log_loss": float(ll_tr),
        "baseline_log_loss": float(base_ll),
        "n_train_snaps": int(len(y_tr)),
        "n_train_players": int(n_tr_pid),
        "n_cal_snaps": int(len(y_ca)),
        "n_cal_players": int(n_ca_pid),
        "n_val_snaps": int(len(y_va)),
        "n_val_players": int(n_va_pid),
        "debut_window": (args.debut_min, args.debut_max),
        "model_type": "HistGradientBoostingClassifier",
        "train_mode": "all_predebut_snapshots",
        "val_players_path": args.val_players,
        "cal_players_path": args.fit_players,
    }
    with open(args.out, "wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out}")
    print(f"  train_log_loss={ll_tr:.4f}  baseline_log_loss={base_ll:.4f}  "
          f"improvement={(base_ll - ll_tr):.4f} nats/sample")


if __name__ == "__main__":
    main()
