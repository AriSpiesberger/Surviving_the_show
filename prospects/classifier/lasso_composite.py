"""Lasso composite buy-score on top of v1.14b hazard probabilities.

Row-per-(player, snap) setup. Each training row = one (player, snap)
prediction moment. Three categories of input features, matching how
we'd score a live player:

  1. Hazards (what we expect to occur in the future):
       p_TOP_100_PROSPECT, p_MLB_DEBUT, p_ESTABLISHED_MLB, p_STAR_PLUS_ELITE
  2. Age (how old the player is):
       age_at_snap_centered (centered around 22)
  3. Observation depth (how much data went into the hazards):
       years_in_pro (= snap_offset)

Target per row:
  3*realized_TOP_100 + 3*realized_MLB_DEBUT
  + 2*realized_ESTABLISHED + 10*realized_STAR_PLUS_ELITE
  (using each row's "realized in (snap, observe_through]" column)

CV: GroupKFold by player_id so the same player's snaps stay in one fold.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict

import numpy as np
from scipy import stats as sstats
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


EVENTS = ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "STAR_PLUS_ELITE")
TARGET_WEIGHTS = {
    "TOP_100_PROSPECT": 3,
    "MLB_DEBUT": 5,
    "ESTABLISHED_MLB": 2,
    "STAR_PLUS_ELITE": 10,
}
AGE_CENTER = 22
YEARS_IN_PRO_CENTER = 3  # center yip for interaction terms


def load_long(path: str) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8", errors="replace")))


def load_birth_dates(db_path: str) -> dict[str, str]:
    """Pull birth_date from prospects table, keyed by player_id."""
    import sqlite3
    out: dict[str, str] = {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute("SELECT player_id, birth_date FROM prospects"):
            if r["birth_date"]:
                out[r["player_id"]] = r["birth_date"]
    return out


def _age_at(birth_iso: str, year: int) -> float | None:
    try:
        bd_y = int(str(birth_iso)[:4])
        return float(year - bd_y)
    except Exception:
        return None


def build_feature_matrix(rows, birth_by_pid: dict[str, str]):
    """One feature vector per (player, snap_offset) row.

    Features (in order):
      0..3  : p_<EVENT>             (4 hazards from the snap)
      4     : age_at_snap_centered  (age - 22)
      5     : years_in_pro          (= snap_offset)

    Returns X, y, groups, meta, feature_names.
    """
    feature_names = (
        [f"p_{ev}" for ev in EVENTS]
        + ["age_at_snap_centered", "years_in_pro"]
        + [f"p_{ev}_x_yip_centered" for ev in EVENTS]
    )
    Xs, ys, groups, metas = [], [], [], []
    skipped_no_age = 0
    for r in rows:
        pid = r["player_id"]
        snap_year = int(r["snap_year"])
        snap_offset = int(r["snap_offset"])
        birth = birth_by_pid.get(pid)
        if not birth:
            skipped_no_age += 1
            continue
        age = _age_at(birth, snap_year)
        if age is None:
            skipped_no_age += 1
            continue
        p_top = float(r["p_TOP_100_PROSPECT"])
        p_mlb = float(r["p_MLB_DEBUT"])
        p_est = float(r["p_ESTABLISHED_MLB"])
        p_starplus = float(r["p_STAR_PLUS_ELITE"])
        yip_c = snap_offset - YEARS_IN_PRO_CENTER
        feat = [
            p_top, p_mlb, p_est, p_starplus,
            age - AGE_CENTER,
            snap_offset,
            p_top * yip_c, p_mlb * yip_c,
            p_est * yip_c, p_starplus * yip_c,
        ]
        # Target: binary weights OR linear time-decay
        def _decay(ename, realized, trigger_col):
            if not realized: return 0.0
            H = TIME_DECAY.get(ename)
            if H is None:
                return float(TARGET_WEIGHTS.get(ename, 0))
            trig = r.get(trigger_col, "")
            if not trig: return 0.0
            try:
                yrs = int(trig) - snap_year
            except Exception:
                return 0.0
            return max(0.0, H - yrs)

        target = (
            _decay("TOP_100_PROSPECT", int(r["realized_TOP_100_PROSPECT"]),
                   "trigger_TOP_100_PROSPECT")
            + _decay("MLB_DEBUT", int(r["realized_MLB_DEBUT"]),
                     "trigger_MLB_DEBUT")
            + _decay("ESTABLISHED_MLB", int(r["realized_ESTABLISHED_MLB"]),
                     "trigger_ESTABLISHED_MLB")
            + _decay("STAR_PLUS_ELITE", int(r["realized_STAR_PLUS_ELITE"]),
                     "trigger_STAR_PLUS_ELITE")
        )
        Xs.append(feat)
        ys.append(target)
        groups.append(pid)
        metas.append({
            "player_id": pid,
            "name": r.get("name", ""),
            "entry_year": int(r["entry_year"]),
            "snap_offset": snap_offset,
            "snap_year": snap_year,
            "bucket": r.get("bucket", ""),
            "years_fwd": int(r["years_fwd"]),
            "age": age,
        })
    if skipped_no_age:
        print(f"  skipped {skipped_no_age:,} rows lacking birth_date")
    return (np.array(Xs, dtype=np.float64),
            np.array(ys, dtype=np.float64),
            np.array(groups),
            metas,
            feature_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", default="val_v14b_long.csv")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--max-entry-year", type=int, default=2022)
    ap.add_argument("--min-years-fwd", type=int, default=3,
                    help="Require this many forward observation years for "
                         "a row to be a training example (default 3).")
    ap.add_argument("--target-weights",
                    default="TOP_100_PROSPECT=3,MLB_DEBUT=5,"
                            "ESTABLISHED_MLB=2,STAR_PLUS_ELITE=10",
                    help="Comma-separated weights, e.g. "
                         "'TOP_100_PROSPECT=1,MLB_DEBUT=1'. Events not "
                         "listed get weight 0.")
    ap.add_argument("--require-eligible",
                    default="",
                    help="Comma-separated event names; only include rows "
                         "where ALL these events are still eligible (event "
                         "has NOT fired by snap). E.g. 'TOP_100_PROSPECT' "
                         "filters to the 'not yet on Top-100' cohort.")
    ap.add_argument("--time-decay",
                    default="",
                    help="If set, use linear time-decay target instead of "
                         "simple binary. Format: 'EVENT=H,EVENT=H' where H "
                         "is the horizon cap. E.g. "
                         "'TOP_100_PROSPECT=3,MLB_DEBUT=4'. Target per event "
                         "= max(0, H - years_to_fire) if fired, else 0.")
    ap.add_argument("--out-prefix", default="lasso_composite_v14b")
    args = ap.parse_args()

    # Parse target weights
    global TARGET_WEIGHTS
    parsed = {}
    for pair in args.target_weights.split(","):
        pair = pair.strip()
        if not pair: continue
        k, v = pair.split("=")
        parsed[k.strip()] = float(v.strip())
    TARGET_WEIGHTS = parsed
    print(f"Target weights: {TARGET_WEIGHTS}")

    require_eligible = [s.strip() for s in args.require_eligible.split(",")
                        if s.strip()]
    if require_eligible:
        print(f"Filter to rows eligible for: {require_eligible}")

    # Parse time-decay
    global TIME_DECAY
    TIME_DECAY = {}
    if args.time_decay:
        for pair in args.time_decay.split(","):
            pair = pair.strip()
            if not pair: continue
            k, v = pair.split("=")
            TIME_DECAY[k.strip()] = float(v.strip())
        print(f"Time-decay horizons: {TIME_DECAY}")

    print(f"Loading {args.long}")
    rows = load_long(args.long)
    rows = [r for r in rows
            if int(r["entry_year"]) <= args.max_entry_year
            and int(r["years_fwd"]) >= args.min_years_fwd]
    if require_eligible:
        rows = [r for r in rows
                if all(int(r.get(f"eligible_{ev}", 0)) == 1
                       for ev in require_eligible)]
    print(f"  filtered: {len(rows):,} rows "
          f"(entry<={args.max_entry_year}, years_fwd>={args.min_years_fwd}"
          f"{', eligible='+str(require_eligible) if require_eligible else ''})")

    print(f"Loading birth dates from {args.db}")
    birth_by_pid = load_birth_dates(args.db)
    print(f"  {len(birth_by_pid):,} prospects have birth_date")

    X, y, groups, metas, feat_names = build_feature_matrix(rows, birth_by_pid)
    n_players = len(set(groups))
    print(f"  X shape: {X.shape}, n_players={n_players:,}")
    print(f"  target mean: {y.mean():.3f}, max: {y.max():.1f}, "
          f"frac > 0: {(y>0).mean():.3f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Player-grouped 5-fold CV so the same player's snaps stay together.
    # Materialize splits to a list so LassoCV is pickleable after fit.
    gkf = GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(X_scaled, y, groups))
    alphas = np.logspace(-4, 0, 50)
    lasso = LassoCV(cv=cv_splits, alphas=alphas,
                    max_iter=20000, n_jobs=-1)
    lasso.fit(X_scaled, y)
    # Clear cv to None before pickling (it's just int splits, not needed
    # after fit and the cv list of ndarrays bloats the pickle).
    lasso.cv = None

    print(f"\nLassoCV (player-grouped 5-fold): best alpha = {lasso.alpha_:.5f}")
    cv_r2 = 1 - lasso.mse_path_.mean(axis=1).min() / y.var()
    print(f"  CV R^2: {cv_r2:.4f}")
    pred = lasso.predict(X_scaled)
    insample_r2 = r2_score(y, pred)
    spear, spp = sstats.spearmanr(pred, y)
    print(f"  in-sample R^2: {insample_r2:.4f}")
    print(f"  Spearman rho(pred, target): {spear:.4f} (p={spp:.2e})")

    print(f"\nNon-zero coefficients (intercept={lasso.intercept_:.3f}):")
    coefs = lasso.coef_
    order = np.argsort(-np.abs(coefs))
    for idx in order:
        if abs(coefs[idx]) < 1e-6: continue
        print(f"  {feat_names[idx]:<42} {coefs[idx]:+.4f}")

    # Decile calibration
    print(f"\nDecile calibration (predicted score vs mean realized target):")
    df = list(zip(pred, y, metas))
    df.sort(key=lambda x: -x[0])
    n_per = max(1, len(df) // 10)
    print(f"  {'decile':<7} {'n':>5} {'pred_mean':>9} {'real_mean':>9} "
          f"{'pct_any_event':>14}")
    for d in range(10):
        chunk = df[d*n_per:(d+1)*n_per] if d < 9 else df[9*n_per:]
        if not chunk: continue
        p_mean = np.mean([x[0] for x in chunk])
        r_mean = np.mean([x[1] for x in chunk])
        any_ev = np.mean([1.0 if x[1] > 0 else 0.0 for x in chunk])
        print(f"  {d+1:<7} {len(chunk):>5d} {p_mean:>9.2f} {r_mean:>9.2f} "
              f"{100*any_ev:>13.1f}%")

    # Per-years_in_pro R^2 (sanity check: predictions improve with depth)
    print(f"\nPer-years_in_pro R^2 (observation depth -> prediction quality):")
    by_yip: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for p, t, m in df:
        by_yip[m["snap_offset"]].append((p, t))
    print(f"  {'yip':>3} {'n':>5} {'R^2':>7} {'Spearman':>10}")
    for yip in sorted(by_yip):
        pairs = by_yip[yip]
        if len(pairs) < 20: continue
        ps = np.array([a for a, _ in pairs]); ts = np.array([b for _, b in pairs])
        try:
            r2 = r2_score(ts, ps)
            sp, _ = sstats.spearmanr(ps, ts)
        except Exception:
            r2 = float("nan"); sp = float("nan")
        print(f"  {yip:>3d} {len(pairs):>5d} {r2:>+7.3f} {sp:>+10.3f}")

    # Save artifacts
    art = {
        "scaler": scaler,
        "lasso": lasso,
        "feature_names": feat_names,
        "target_weights": TARGET_WEIGHTS,
        "events": list(EVENTS),
        "age_center": AGE_CENTER,
        "max_entry_year_train": args.max_entry_year,
        "min_years_fwd_train": args.min_years_fwd,
    }
    pkl_path = f"{args.out_prefix}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(art, f)
    print(f"\nSaved {pkl_path}")

    score_path = f"{args.out_prefix}_train_scores.csv"
    with open(score_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "name", "entry_year", "bucket",
                    "snap_offset", "snap_year", "years_fwd",
                    "score", "target"])
        for p, t, m in zip(pred, y, metas):
            w.writerow([m["player_id"], m["name"], m["entry_year"],
                        m["bucket"], m["snap_offset"], m["snap_year"],
                        m["years_fwd"], f"{p:.4f}", f"{t:.1f}"])
    print(f"Saved {score_path}")


if __name__ == "__main__":
    main()
