"""Train v1.18b production artifacts: landmark-trained hazards + refit
v1.18 downstream (L1-logistic bundle + time-to-debut regression).

v1.18b mirrors v1.18 in its downstream artifacts (lasso_logits + time_to_debut)
but changes the UPSTREAM hazards from the contemporaneous v1.17 prod model to
landmark-trained hazards via `prospects.classifier.architectures.landmark_survival`.

Pipeline (six stages):
  1. Build landmark panel (one row per (player, landmark S)).
  2. Fit per-event landmark HistGBT hazards on 100% of landmarks ->
       models/event_classifiers_v1.18b_landmark_prod.pkl
  3. Score the v1.17 fit pids slice with landmark hazards ->
       results/training/v1.18b_landmark_fit_long.csv
       (same column shape as v1.17_prod_fit_long.csv so the v1.18 trainers
        consume it without modification.)
  4. Same for val ->
       results/training/v1.18b_landmark_val_long.csv
  5. Combine, train L1-logistic bundle (fit_lasso_logits_v18) ->
       models/lasso_logits_v1.18b_prod.pkl
  6. Train time-to-debut Lasso (fit_time_to_debut_v18) ->
       models/time_to_debut_v1.18b_prod.pkl

v1.18 stays untouched: this script writes _v1.18b_ artifacts side-by-side
so we can score the same prospects with both heads for head-to-head
validation.

Usage:
    python -m scripts_v17.train.train_v1_18b_prod

    # Reuse a previously-built landmark panel + hazards (skip retraining):
    python -m scripts_v17.train.train_v1_18b_prod --skip-hazards

    # Skip scoring (use existing landmark fit/val CSVs):
    python -m scripts_v17.train.train_v1_18b_prod --skip-score

    # Skip downstream refit (just produce the long CSVs):
    python -m scripts_v17.train.train_v1_18b_prod --skip-downstream
"""
from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures import landmark_survival as lm
from prospects.classifier.architectures.survival import (
    ELITE_KEY, MAX_OBS_YEAR, STAR_KEY, _trigger_year,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


HAZ_OUT = REPO_ROOT / "models" / "event_classifiers_v1.18b_landmark_prod.pkl"
LANDMARK_PANEL_CACHE = REPO_ROOT / "results" / "training" / "landmark_panel_v1.18b.npz"
LM_FIT_LONG = REPO_ROOT / "results" / "training" / "v1.18b_landmark_fit_long.csv"
LM_VAL_LONG = REPO_ROOT / "results" / "training" / "v1.18b_landmark_val_long.csv"
LM_COMBINED_LONG = REPO_ROOT / "results" / "training" / "v1.18b_landmark_all_long.csv"
BUNDLE_OUT = REPO_ROOT / "models" / "lasso_logits_v1.18b_prod.pkl"
TIMING_OUT = REPO_ROOT / "models" / "time_to_debut_v1.18b_prod.pkl"
FIT_PIDS = REPO_ROOT / "results" / "training" / "v17_prod_fit_pids.txt"
VAL_PIDS = REPO_ROOT / "results" / "training" / "v17_prod_val_pids.txt"


def _ev_name(e) -> str:
    return e.name if hasattr(e, "name") else str(e).lstrip("_")


def _entry_year(player: dict, stats_by_pid: dict) -> int | None:
    dy = player.get("draft_year")
    is_intl = int(player.get("is_international") or 0)
    if dy is not None and not is_intl:
        return int(dy)
    yrs = [s.get("season_year")
           for s in stats_by_pid.get(player["player_id"], [])
           if s.get("season_year") is not None
           and (s.get("level") or "").upper() != "MLB"]
    if yrs:
        return int(min(yrs))
    if dy is not None:
        return int(dy)
    return None


def _bucket_of(player: dict) -> str:
    if int(player.get("is_international") or 0) == 1:
        return "IFA"
    r = player.get("draft_round")
    if r is None:
        return "IFA"
    r = int(r)
    if r == 1:
        return "R1"
    if r <= 3:
        return "R2-R3"
    if r <= 10:
        return "R4-R10"
    return "R10+"


def score_pids_with_landmark(
    hazards: dict, prospects_all: list[dict], stats_by_pid: dict,
    pid_set: set[str], out_csv: Path,
    max_entry_year: int = 2020, observe_through: int = MAX_OBS_YEAR,
    max_offset: int = 10, horizon: int = 15, verbose: bool = True,
) -> int:
    """For each pid in pid_set, score at snap = entry_year + 0..max_offset
    using landmark hazards. Emits CSV with same columns as
    v1.17_prod_fit_long.csv so fit_lasso_logits_v18 / fit_time_to_debut_v18
    consume it unchanged.

    Returns number of rows written."""
    cohort = [p for p in prospects_all if p["player_id"] in pid_set]
    # Filter cohort to those with viable entry_year <= max_entry_year (mirrors
    # contemporaneous score_v14c_cal_slice_raw semantics).
    enriched: list[dict] = []
    for r in cohort:
        ent = _entry_year(r, stats_by_pid)
        if ent is None or ent > max_entry_year:
            continue
        rc = dict(r)
        rc["_entry_year"] = ent
        rc["_bucket"] = _bucket_of(r)
        enriched.append(rc)

    if verbose:
        print(f"[score] cohort={len(enriched):,} / {len(cohort):,} after "
              f"entry<={max_entry_year} filter", flush=True)

    # Group (player, snap) pairs by snap so we can batch-score per snap.
    snap_groups: dict[int, list[dict]] = {}
    for r in enriched:
        ent = r["_entry_year"]
        debut = r.get("mlb_debut_year")
        for off in range(0, max_offset + 1):
            snap = ent + off
            if snap > observe_through:
                break
            if debut is not None and debut <= snap:
                # Already debuted by snap — drop (mirrors score_v14c logic).
                continue
            rc = dict(r)
            rc["_snap"] = snap
            rc["_offset"] = off
            snap_groups.setdefault(snap, []).append(rc)

    # Score per snap_year batch, accumulate rows.
    event_keys = [k for k in hazards if not isinstance(k, str)
                  or k in (ELITE_KEY, STAR_KEY)]
    out_rows: list[dict] = []
    snap_keys = sorted(snap_groups.keys())
    for si, snap in enumerate(snap_keys):
        group = snap_groups[snap]
        sub_stats = {
            r["player_id"]: [s for s in stats_by_pid.get(r["player_id"], [])
                             if (s.get("season_year") or 0) <= snap]
            for r in group
        }
        out = lm.predict_cumulative_batch_landmark(
            hazards, group, sub_stats,
            current_year=snap, horizon=horizon,
        )
        for i, r in enumerate(group):
            row = {
                "player_id":         r["player_id"],
                "name":              r.get("name"),
                "draft_year":        r.get("draft_year"),
                "draft_round":       r.get("draft_round"),
                "is_international":  int(r.get("is_international") or 0),
                "bucket":            r["_bucket"],
                "entry_year":        r["_entry_year"],
                "snap_year":         snap,
                "snap_offset":       r["_offset"],
                "years_fwd":         observe_through - snap,
                "mlb_debut_year":    r.get("mlb_debut_year"),
            }
            per_ev = {}
            for e in event_keys:
                ename = _ev_name(e)
                p_cal = float(out[e][i])
                trig = _trigger_year(r, e)
                eligible = int(trig is None or trig > snap)
                realized = int(trig is not None and trig > snap
                               and trig <= observe_through)
                per_ev[ename] = (p_cal, trig, eligible, realized)
                row[f"p_{ename}"] = p_cal
                row[f"eligible_{ename}"] = eligible
                row[f"realized_{ename}"] = realized
                row[f"trigger_{ename}"] = trig
                # Timing moments from the landmark inference. The cumulative
                # alone loses temporal shape (every "high-cumP" player looks
                # alike); mean_t/sd_t carry the model's explicit when-prediction.
                mt = out.get(("mean_t", e))
                st = out.get(("sd_t", e))
                if mt is not None:
                    row[f"mean_t_{ename}"] = float(mt[i])
                if st is not None:
                    row[f"sd_t_{ename}"] = float(st[i])
            # STAR_PLUS_ELITE union (mirror score_v14c_cal_slice_raw).
            if "STAR" in per_ev and "ELITE" in per_ev:
                ps, ts, _, _ = per_ev["STAR"]
                pe, te, _, _ = per_ev["ELITE"]
                p_u = 1.0 - (1.0 - ps) * (1.0 - pe)
                trigs = [t for t in (ts, te) if t is not None]
                trig_u = min(trigs) if trigs else None
                elig_u = int(trig_u is None or trig_u > snap)
                real_u = int(trig_u is not None and trig_u > snap
                             and trig_u <= observe_through)
                row["p_STAR_PLUS_ELITE"] = p_u
                row["eligible_STAR_PLUS_ELITE"] = elig_u
                row["realized_STAR_PLUS_ELITE"] = real_u
                row["trigger_STAR_PLUS_ELITE"] = trig_u
            out_rows.append(row)
        if verbose and (si % 5 == 0 or si == len(snap_keys) - 1):
            print(f"  [score] snap={snap}  group={len(group)}  "
                  f"({si+1}/{len(snap_keys)})", flush=True)

    if not out_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("")
        return 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fnames = list(out_rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        w.writerows(out_rows)
    return len(out_rows)


def _read_pids(p: Path) -> set[str]:
    return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--skip-hazards", action="store_true",
                    help="Reuse the existing landmark hazards pkl + cached panel")
    ap.add_argument("--skip-score", action="store_true",
                    help="Reuse the existing v1.18b_landmark_{fit,val}_long.csv")
    ap.add_argument("--skip-downstream", action="store_true",
                    help="Skip the lasso_logits + time_to_debut refit")
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--min-landmark-year", type=int, default=2007)
    ap.add_argument("--max-landmark-year", type=int, default=MAX_OBS_YEAR - 1)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    ap.add_argument("--max-offset", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()

    db = ProspectDB(args.db)
    t_start = time.time()
    print("=" * 78)
    print("v1.18b PROD TRAIN")
    print("=" * 78)

    # --- 1+2. Landmark panel + hazards ---
    if args.skip_hazards and HAZ_OUT.exists():
        print(f"[1/6] skip-hazards: loading {HAZ_OUT.name}", flush=True)
        with HAZ_OUT.open("rb") as f:
            hazards = pickle.load(f)
        # Need to re-fetch prospects + stats for scoring; cheap relative to
        # the hazard fit.
        print(f"[2/6] re-fetching prospects + stats for scoring", flush=True)
        with db._connect() as conn:
            prospects_all = [dict(r) for r in conn.execute("""
                SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                       o.year_top_100, o.year_top_25,
                       o.year_all_star_once, o.year_all_star_three,
                       o.year_major_award, o.year_hof_trajectory,
                       o.events_json, o.final_mlb_year
                FROM prospects p
                JOIN career_outcomes o ON o.player_id = p.player_id
                WHERE (p.draft_year IS NOT NULL AND p.draft_year <= ?)
                   OR (COALESCE(p.is_international, 0) = 1)
            """, (args.max_draft_year,)).fetchall()]
            stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        stats_by_pid: dict[str, list] = {}
        for s in stats_rows:
            d = dict(s)
            stats_by_pid.setdefault(d["player_id"], []).append(d)
    else:
        print(f"[1/6] Building landmark panel "
              f"(max_draft_year={args.max_draft_year}, "
              f"landmark={args.min_landmark_year}..{args.max_landmark_year})",
              flush=True)
        t0 = time.time()
        X_lm, pids, S_yrs, joined, stats_by_pid = lm.build_landmark_panel(
            db,
            max_draft_year=args.max_draft_year,
            min_landmark_year=args.min_landmark_year,
            max_landmark_year=args.max_landmark_year,
            include_ifa=True,
            verbose=True,
        )
        print(f"  panel built in {time.time()-t0:.0f}s  "
              f"X_lm={X_lm.shape}", flush=True)

        print(f"\n[2/6] Fitting landmark hazards (100% of landmarks, prod)",
              flush=True)
        t0 = time.time()
        hazards = lm.fit_landmark_hazards(
            X_lm, joined, S_yrs, stats_by_pid,
            train_mask=None, seed=42, verbose=True,
        )
        print(f"  hazards trained in {time.time()-t0:.0f}s", flush=True)

        HAZ_OUT.parent.mkdir(parents=True, exist_ok=True)
        with HAZ_OUT.open("wb") as f:
            pickle.dump(hazards, f)
        print(f"  saved {HAZ_OUT}", flush=True)
        # We still need prospects_all + stats for scoring; reuse what we have.
        prospects_all = [dict(p) for p in joined]
        # joined may carry duplicate dicts (one per landmark) — dedupe by pid.
        seen = set()
        dedup = []
        for p in prospects_all:
            if p["player_id"] in seen:
                continue
            seen.add(p["player_id"])
            dedup.append(p)
        prospects_all = dedup
        del X_lm, joined, S_yrs, pids
        import gc; gc.collect()

    # --- 3+4. Score fit + val slices ---
    if args.skip_score and LM_FIT_LONG.exists() and LM_VAL_LONG.exists():
        print(f"\n[3+4/6] skip-score: reusing {LM_FIT_LONG.name} + "
              f"{LM_VAL_LONG.name}", flush=True)
    else:
        for src_pids, out_csv, label in (
            (FIT_PIDS, LM_FIT_LONG, "fit"),
            (VAL_PIDS, LM_VAL_LONG, "val"),
        ):
            if not src_pids.exists():
                sys.exit(f"FATAL: missing pids file {src_pids}")
            pid_set = _read_pids(src_pids)
            print(f"\n[{'3' if label=='fit' else '4'}/6] Scoring {label} slice "
                  f"({len(pid_set):,} pids) with landmark hazards", flush=True)
            t0 = time.time()
            n = score_pids_with_landmark(
                hazards, prospects_all, stats_by_pid, pid_set, out_csv,
                max_entry_year=args.max_entry_year,
                observe_through=MAX_OBS_YEAR,
                max_offset=args.max_offset, horizon=args.horizon,
                verbose=True,
            )
            print(f"  wrote {n:,} rows in {time.time()-t0:.0f}s -> {out_csv}",
                  flush=True)

    # --- 5+6. Downstream refit ---
    if args.skip_downstream:
        print(f"\n[5+6/6] skip-downstream: leaving v1.18b downstream alone",
              flush=True)
        print(f"\n=== v1.18b TRAIN DONE (skipped downstream) in "
              f"{time.time()-t_start:.0f}s ===")
        return

    # Combine fit + val for the bundle/timing trainers (they were designed for
    # combined input — see train_v1_18_prod.py).
    print(f"\n[5/6] Combining fit + val long for downstream refit",
          flush=True)
    import pandas as pd
    fit_df = pd.read_csv(LM_FIT_LONG)
    val_df = pd.read_csv(LM_VAL_LONG)
    combined = pd.concat([fit_df, val_df], ignore_index=True)
    LM_COMBINED_LONG.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(LM_COMBINED_LONG, index=False)
    print(f"  combined: {len(combined):,} rows -> {LM_COMBINED_LONG}",
          flush=True)

    # Train L1-logistic bundle on landmark hazards.
    bundle_tmp = str(BUNDLE_OUT) + ".tmp"
    print(f"\n[5/6 ctd] Refitting lasso_logits bundle -> {BUNDLE_OUT.name}",
          flush=True)
    rc = subprocess.run(
        [sys.executable, "-m", "scripts_v17.train.fit_lasso_logits_v18",
         "--fit", str(LM_COMBINED_LONG),
         "--val", str(LM_COMBINED_LONG),
         "--db",  args.db,
         "--out", bundle_tmp],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        sys.exit(rc)
    shutil.move(bundle_tmp, BUNDLE_OUT)
    print(f"  wrote {BUNDLE_OUT}", flush=True)

    # Train time-to-debut Lasso on landmark hazards. Use the v18b trainer
    # which extends features with mean_t / sd_t timing moments emitted by
    # predict_cumulative_batch_landmark (the cumulative alone loses near-
    # term timing signal compared to v1.18's LOCF-bias-as-oracle).
    timing_tmp = str(TIMING_OUT) + ".tmp"
    print(f"\n[6/6] Refitting time_to_debut (v1.18b w/ mean_t) -> "
          f"{TIMING_OUT.name}", flush=True)
    rc = subprocess.run(
        [sys.executable, "-m", "scripts_v17.train.fit_time_to_debut_v18b",
         "--fit", str(LM_COMBINED_LONG),
         "--val", str(LM_COMBINED_LONG),
         "--db",  args.db,
         "--bundle", str(BUNDLE_OUT),
         "--include-p-debut",
         "--out", timing_tmp],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        sys.exit(rc)
    shutil.move(timing_tmp, TIMING_OUT)
    print(f"  wrote {TIMING_OUT}", flush=True)

    print(f"\n=== v1.18b TRAIN COMPLETE in "
          f"{(time.time()-t_start)/60:.1f} min ===")
    print(f"  hazards:  {HAZ_OUT}")
    print(f"  bundle:   {BUNDLE_OUT}")
    print(f"  timing:   {TIMING_OUT}")
    print(f"  longs:    {LM_FIT_LONG.name}, {LM_VAL_LONG.name}")


if __name__ == "__main__":
    main()
