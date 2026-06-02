"""Honest validation for Model B on the held-out 10% val slice.

Reports per-class AUC, Brier, reliability deciles, multinomial log-loss,
and the confusion matrix of argmax-predicted class vs realized.

Outputs to results/<prefix>_<YYYY-MM-DD>/:
    report.txt          consolidated stdout
    per_class.csv       per-class AUC/Brier/log-loss/base rate
    reliability.csv     per-class decile reliability table
    confusion.csv       predicted-argmax × actual count
    val_preds.csv       per-player predicted probs + actual label

Usage:
    python -m scripts_v17.validate.validate_model_b \\
        --model models/model_b_outcomes_v1.17h.pkl \\
        --panel panel_v1.17.npz \\
        --out-prefix val_model_b_v17h
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from prospects.features.scouting import N_FEATURES
from prospects.storage import ProspectDB

from scripts_v17.train.fit_model_b_honest import (
    CLASSES, label_player, _all_predebut_idx, _latest_predebut_idx,
    _load_all_stats, _load_pids,
)


class _Tee:
    def __init__(self, *s): self._s = s
    def write(self, x):
        for s in self._s: s.write(x)
    def flush(self):
        for s in self._s: s.flush()


def _apply_calibrators(p_raw: np.ndarray, calibrators) -> np.ndarray:
    """Per-class one-vs-rest Platt; renormalize across classes."""
    out = np.zeros_like(p_raw)
    for k, lr_k in enumerate(calibrators):
        if lr_k is None:
            out[:, k] = p_raw[:, k]
        else:
            out[:, k] = lr_k.predict_proba(p_raw[:, k:k+1])[:, 1]
    # Renormalize so each row sums to 1
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to model_b_outcomes_v*.pkl (honest)")
    ap.add_argument("--panel", default="panel_v1.17.npz")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--val-players", default=None,
                    help="Override val_players path (default: read from "
                         "model pkl's val_players_path)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    print(f"Loading model {args.model}")
    with open(args.model, "rb") as fh:
        mb = pickle.load(fh)
    clf = mb["model"]; scaler = mb["scaler"]
    cal_calibrators = mb.get("cal_calibrators", [None] * len(CLASSES))
    val_path = args.val_players or mb["val_players_path"]
    debut_min, debut_max = mb.get("debut_window", (2010, 2024))
    print(f"  classes={CLASSES}  debut_window={debut_min}-{debut_max}")
    print(f"  val_players={val_path}")

    val_pids = _load_pids(val_path)
    print(f"  {len(val_pids):,} val players")

    print(f"Loading panel {args.panel}")
    with np.load(args.panel, allow_pickle=True) as d:
        X_full = d["X"].astype(np.float32, copy=False)
        pids = np.asarray(d["pids"])
        years = np.asarray(d["years"], dtype=int)
    assert X_full.shape[1] == N_FEATURES

    print(f"Loading debut years from {args.db}")
    db = ProspectDB(args.db)
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT player_id, mlb_debut_year FROM career_outcomes "
            "WHERE mlb_debut_year IS NOT NULL").fetchall()
    debut_by_pid = {r["player_id"]: int(r["mlb_debut_year"]) for r in rows
                    if debut_min <= int(r["mlb_debut_year"]) <= debut_max
                    and r["player_id"] in val_pids}
    print(f"  val debutees {debut_min}-{debut_max}: {len(debut_by_pid):,}")

    earliest = _earliest_predebut_idx(pids, years, debut_by_pid)
    print(f"  matched to panel rows: {len(earliest):,}")

    stats = _load_all_stats(args.db)
    stats_by_pid = {pid: g for pid, g in stats.groupby("player_id")}

    val_idx, val_y, val_pid_list = [], [], []
    for pid, idx in earliest.items():
        debut = debut_by_pid[pid]
        post = stats_by_pid.get(pid)
        # No stat rows at all → default to 'debut' (no quality stats, no
        # demotion evidence).
        label = "debut" if post is None else label_player(post, debut)
        val_idx.append(idx); val_y.append(CLASSES.index(label))
        val_pid_list.append(pid)
    X_va = np.nan_to_num(X_full[np.asarray(val_idx)], nan=0.0,
                         posinf=0.0, neginf=0.0)
    y_va = np.asarray(val_y, dtype=int)

    date = _dt.date.today().isoformat()
    out_dir = os.path.join(args.results_dir, f"{args.out_prefix}_{date}")
    os.makedirs(out_dir, exist_ok=True)
    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, buf)

    print(f"validate_model_b.py  prefix={args.out_prefix}  out_dir={out_dir}")
    print(f"  model={args.model}  date={date}")
    print(f"  panel={args.panel}  val_players={val_path}")
    print(f"  val debutees scored: {len(y_va):,}\n")

    print("Class distribution (val):")
    for k, name in enumerate(CLASSES):
        n = int((y_va == k).sum())
        print(f"  {name:<10s} {n:>4d}  {n/len(y_va):.1%}")

    p_raw = clf.predict_proba(scaler.transform(X_va))
    p_cal = _apply_calibrators(p_raw, cal_calibrators)

    # ---- Per-class metrics (raw and calibrated)
    print(f"\n{'='*72}\nPER-CLASS METRICS (val slice, held out)\n{'='*72}")
    print(f"{'class':<10} {'base':>7} {'AUC':>7} {'Brier_raw':>10} "
          f"{'Brier_cal':>10} {'LL_raw':>8} {'LL_cal':>8}")
    per_class_rows = []
    for k, name in enumerate(CLASSES):
        y_k = (y_va == k).astype(int)
        base = float(y_k.mean())
        try:
            auc = float(roc_auc_score(y_k, p_raw[:, k])) if 0 < y_k.sum() < len(y_k) else float("nan")
        except Exception:
            auc = float("nan")
        b_raw = float(brier_score_loss(y_k, p_raw[:, k]))
        b_cal = float(brier_score_loss(y_k, p_cal[:, k]))
        ll_raw = float(-(y_k * np.log(p_raw[:, k].clip(1e-9, 1)) +
                         (1 - y_k) * np.log((1 - p_raw[:, k]).clip(1e-9, 1))).mean())
        ll_cal = float(-(y_k * np.log(p_cal[:, k].clip(1e-9, 1)) +
                         (1 - y_k) * np.log((1 - p_cal[:, k]).clip(1e-9, 1))).mean())
        print(f"{name:<10} {base:>7.1%} {auc:>7.3f} {b_raw:>10.4f} "
              f"{b_cal:>10.4f} {ll_raw:>8.4f} {ll_cal:>8.4f}")
        per_class_rows.append({
            "class": name, "n_pos": int(y_k.sum()), "base_rate": base,
            "auc": auc, "brier_raw": b_raw, "brier_cal": b_cal,
            "logloss_raw": ll_raw, "logloss_cal": ll_cal,
        })
    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(out_dir, "per_class.csv"), index=False)

    # Multinomial log-loss
    mll_raw = float(-np.log(p_raw[np.arange(len(y_va)), y_va] + 1e-9).mean())
    mll_cal = float(-np.log(p_cal[np.arange(len(y_va)), y_va] + 1e-9).mean())
    base = np.array([(y_va == k).mean() for k in range(len(CLASSES))])
    mll_base = float(-np.log(base[y_va] + 1e-9).mean())
    print(f"\nmultinomial log-loss   raw={mll_raw:.4f}  cal={mll_cal:.4f}  "
          f"baseline={mll_base:.4f}")
    print(f"  improvement vs baseline: raw={mll_base - mll_raw:+.4f}  "
          f"cal={mll_base - mll_cal:+.4f} nats/sample")

    # ---- Reliability decile bins per class (calibrated)
    print(f"\n{'='*72}\nRELIABILITY (calibrated, val slice, decile bins)\n{'='*72}")
    rel_rows = []
    for k, name in enumerate(CLASSES):
        p = p_cal[:, k]
        y_k = (y_va == k).astype(int)
        order = np.argsort(p)
        n = len(p)
        print(f"\n  {name} (base rate {y_k.mean():.1%}):")
        print(f"    {'decile':>6} {'n':>4} {'pred%':>7} {'obs%':>7}")
        for d in range(10):
            lo = d * n // 10
            hi = (d + 1) * n // 10
            idx = order[lo:hi]
            if len(idx) == 0:
                continue
            pred = float(p[idx].mean())
            obs = float(y_k[idx].mean())
            print(f"    {d:>6d} {len(idx):>4d} {pred*100:>6.1f}% "
                  f"{obs*100:>6.1f}%")
            rel_rows.append({"class": name, "decile": d, "n": len(idx),
                             "pred": pred, "obs": obs})
    pd.DataFrame(rel_rows).to_csv(
        os.path.join(out_dir, "reliability.csv"), index=False)

    # ---- Confusion matrix (argmax predicted vs actual)
    y_hat = p_cal.argmax(axis=1)
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, h in zip(y_va, y_hat):
        conf[t, h] += 1
    print(f"\n{'='*72}\nCONFUSION MATRIX (rows=actual, cols=predicted argmax)"
          f"\n{'='*72}")
    print(f"  {'':<10}" + " ".join(f"{c:>10}" for c in CLASSES) + "   total")
    for t, name in enumerate(CLASSES):
        row = conf[t]
        print(f"  {name:<10}" + " ".join(f"{v:>10d}" for v in row) +
              f"   {int(row.sum()):>5d}")
    print(f"  {'pred_tot':<10}" + " ".join(f"{int(conf[:, h].sum()):>10d}"
                                            for h in range(len(CLASSES))))
    overall_acc = float((y_va == y_hat).mean())
    print(f"\n  overall argmax accuracy: {overall_acc:.1%}")
    pd.DataFrame(conf, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(out_dir, "confusion.csv"))

    # ---- Per-player predictions
    pred_df = pd.DataFrame({
        "player_id": val_pid_list,
        "actual": [CLASSES[k] for k in y_va],
        "argmax_pred": [CLASSES[k] for k in y_hat],
    })
    for k, name in enumerate(CLASSES):
        pred_df[f"p_raw_{name}"] = p_raw[:, k]
        pred_df[f"p_cal_{name}"] = p_cal[:, k]
    pred_df.to_csv(os.path.join(out_dir, "val_preds.csv"), index=False)
    print(f"\nsaved per-player preds: {len(pred_df)} rows")

    # ---- Top-decile precision per class (a key business signal)
    print(f"\n{'='*72}\nTOP-DECILE PRECISION (calibrated)\n{'='*72}")
    print(f"{'class':<10} {'k':>5} {'n_pos_in_top':>12} {'precision':>10} "
          f"{'recall':>7} {'lift':>6}")
    for k, name in enumerate(CLASSES):
        p = p_cal[:, k]
        y_k = (y_va == k).astype(int)
        n_top = max(1, int(round(0.10 * len(p))))
        order = np.argsort(-p)[:n_top]
        prec = float(y_k[order].mean()) if n_top else float("nan")
        rec = float(y_k[order].sum() / max(y_k.sum(), 1))
        base = float(y_k.mean())
        lift = prec / base if base > 0 else float("nan")
        print(f"{name:<10} {n_top:>5d} {int(y_k[order].sum()):>12d} "
              f"{prec:>10.1%} {rec:>7.1%} {lift:>5.2f}x")

    sys.stdout = real_stdout
    report = os.path.join(out_dir, "report.txt")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
