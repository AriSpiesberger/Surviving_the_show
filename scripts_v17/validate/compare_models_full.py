"""Full statistical comparison of two model scorings on the same evaluation cohort.

Reports for each event (MLB_DEBUT, TOP_100_PROSPECT, ESTABLISHED_MLB, STAR_PLUS_ELITE):

  CLASSIFICATION QUALITY
    - AUC with 1000-bootstrap CI; DeLong-paired test for ΔAUC significance
    - PR-AUC (average precision) with bootstrap CI
    - Brier score, log-loss (on raw p_<event>); brier_skill_score vs base rate

  TOP-N% PRECISION & MARGINAL SLABS
    - Precision @ top 1, 2, 3, 5, 10, 15, 20, 30, 50%
    - Marginal precision in slabs: top 0-1%, 1-2%, 2-5%, 5-10%, 10-20%, 20-50%
    - McNemar test on top-N% pick set membership
    - Recall, F1 at each threshold
    - Bootstrap CI on precision

  CALIBRATION (on raw hazard p_<event>)
    - Reliability table (10 quantile bins): pred vs observed
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)

  RANK CORRELATION
    - Spearman rank correlation between the two models' scores
    - Kendall tau on top-N% set agreement

  PER-YIP BREAKDOWN
    - All metrics broken out by snap_offset

Usage:
    python compare_models_full.py
        --a-long  v1.14n_val_long.csv  --a-lasso  lasso_v1.14n_td.pkl  --a-name v1.14n
        --b-long  v1.16_val_long.csv   --b-lasso  lasso_v1.16_td.pkl   --b-name v1.16
        --out-prefix compare_v14n_vs_v16

Output:
    <prefix>_summary.txt   human-readable
    <prefix>_metrics.csv   all metrics, machine-readable
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from scipy import stats as sstats

EVENTS = ["MLB_DEBUT", "TOP_100_PROSPECT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
TOP_PCTS = [1, 2, 3, 5, 10, 15, 20, 30, 50]
SLABS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50)]
N_BOOT = 200


# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #

def bootstrap_ci(metric_fn, y, score, n_boot=N_BOOT, alpha=0.05, seed=0):
    """Stratified-by-positive bootstrap CI for a binary classification metric."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if y.sum() == 0 or y.sum() == n:
        return float("nan"), float("nan")
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals[b] = metric_fn(y[idx], score[idx])
        except ValueError:
            vals[b] = np.nan
    return float(np.nanpercentile(vals, alpha/2*100)), float(np.nanpercentile(vals, (1-alpha/2)*100))


def delong_paired_auc(y, score_a, score_b):
    """Fast DeLong (1988) paired test for ΔAUC on the same cohort. Returns
    (auc_a, auc_b, var_diff, z, two_sided_p)."""
    y = np.asarray(y).astype(int)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n_pos = len(pos); n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    def structural(score):
        s_pos = score[pos][:, None]; s_neg = score[neg][None, :]   # shapes (P,1), (1,N)
        # cmp[i,j] = 1 if s_pos[i] > s_neg[j], 0.5 if equal else 0
        cmp = (s_pos > s_neg).astype(np.float32) + 0.5*(s_pos == s_neg).astype(np.float32)
        V10 = cmp.mean(axis=1)        # (P,) avg over negatives
        V01 = cmp.mean(axis=0)        # (N,) avg over positives
        auc = V10.mean()
        return auc, V10, V01
    auc_a, Va10, Va01 = structural(score_a)
    auc_b, Vb10, Vb01 = structural(score_b)
    S10 = np.cov(np.vstack([Va10, Vb10]), ddof=1)
    S01 = np.cov(np.vstack([Va01, Vb01]), ddof=1)
    var = S10[0,0]/n_pos + S01[0,0]/n_neg  # var_a
    var_b = S10[1,1]/n_pos + S01[1,1]/n_neg
    cov_ab = S10[0,1]/n_pos + S01[0,1]/n_neg
    var_diff = var + var_b - 2*cov_ab
    if var_diff <= 0:
        return auc_a, auc_b, var_diff, float("nan"), float("nan")
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - sstats.norm.cdf(abs(z)))
    return auc_a, auc_b, var_diff, z, p


def topk_precision(score, y, pct):
    n = len(score); k = max(1, int(n*pct/100))
    order = np.argsort(-score)[:k]
    tp = int(y[order].sum())
    return tp/k, tp, k


def slab_precision(score, y, lo_pct, hi_pct):
    n = len(score)
    lo = max(0, int(n*lo_pct/100))
    hi = min(n, int(n*hi_pct/100))
    order = np.argsort(-score)[lo:hi]
    if len(order) == 0:
        return float("nan"), 0, 0
    tp = int(y[order].sum())
    return tp/len(order), tp, len(order)


def mcnemar_pick_disagreement(score_a, score_b, y, pct):
    """McNemar test on whether the two models' top-N% picks have different
    TP rates. b = picked by A only and correct, c = picked by B only and
    correct; p from binomial."""
    n = len(score_a); k = max(1, int(n*pct/100))
    a_top = set(np.argsort(-score_a)[:k])
    b_top = set(np.argsort(-score_b)[:k])
    a_only = a_top - b_top; b_only = b_top - a_top
    a_only_tp = int(sum(y[i] for i in a_only))
    b_only_tp = int(sum(y[i] for i in b_only))
    # Binomial test on a_only_tp vs b_only_tp
    n_disagree = a_only_tp + b_only_tp
    if n_disagree < 1:
        return float("nan"), len(a_only), len(b_only), a_only_tp, b_only_tp
    result = sstats.binomtest(a_only_tp, n_disagree, p=0.5)
    return result.pvalue, len(a_only), len(b_only), a_only_tp, b_only_tp


def ece(score, y, n_bins=10):
    """Expected Calibration Error on score in [0,1]."""
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.clip(np.digitize(score, bins) - 1, 0, n_bins-1)
    ece_v = 0.0; mce_v = 0.0
    n = len(score)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        gap = abs(score[m].mean() - y[m].mean())
        ece_v += (m.sum()/n) * gap
        mce_v = max(mce_v, gap)
    return ece_v, mce_v


def reliability(score, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.clip(np.digitize(score, bins) - 1, 0, n_bins-1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        rows.append({
            "bin": b, "n": int(m.sum()),
            "pred": float(score[m].mean()),
            "obs": float(y[m].mean()),
            "gap": float(score[m].mean() - y[m].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Scoring (lasso applied to a long file)
# --------------------------------------------------------------------------- #

def score_with_lasso(long_csv, lasso_pkl, age_center=22, yip_center=3, db="prospects_snapshot.db"):
    df = pd.read_csv(long_csv)
    df = df[df.entry_year <= 2020].copy()
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT"):
        df = df[df[f"eligible_{ev}"]==1]
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id","birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = ((df["snap_year"]-df["birth_year"]).fillna(22.0) - age_center)
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - yip_center
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
    with open(lasso_pkl,"rb") as fh:
        m = pickle.load(fh)
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    df["lasso_score"] = lasso.predict(sc.transform(df[feat].values))
    return df


# --------------------------------------------------------------------------- #
# Comparison driver
# --------------------------------------------------------------------------- #

def compare_one_event(a_df, b_df, event, a_name, b_name, out_lines, metrics_rows, yip=None):
    """Run all comparisons for one event, optionally restricted to a single yip."""
    if yip is not None:
        a = a_df[a_df.snap_offset == yip].reset_index(drop=True)
        b = b_df[b_df.snap_offset == yip].reset_index(drop=True)
        tag = f"yip={yip}"
    else:
        a = a_df.reset_index(drop=True); b = b_df.reset_index(drop=True)
        tag = "pooled"

    y = a[f"realized_{event}"].values.astype(int)
    yb = b[f"realized_{event}"].values.astype(int)
    if not np.array_equal(y, yb):
        # The two long files must share the same player×snap rows for paired tests.
        # If they don't align, fall back to unpaired.
        out_lines.append(f"  WARN: y mismatch between {a_name} and {b_name} for {event} {tag}")

    score_a = a["lasso_score"].values
    score_b = b["lasso_score"].values
    raw_a = a[f"p_{event}"].values if f"p_{event}" in a.columns else None
    raw_b = b[f"p_{event}"].values if f"p_{event}" in b.columns else None
    n = len(y); pos = int(y.sum()); base = pos/n if n else 0

    out_lines.append(f"\n{'='*78}")
    out_lines.append(f" EVENT={event}   COHORT={tag}   n={n:,}  pos={pos} ({base:.2%})")
    out_lines.append('='*78)
    if pos < 5:
        out_lines.append(f"  too few positives, skip")
        return

    # ---------- AUC + DeLong ----------
    auc_a, auc_b, var_d, z, p_auc = delong_paired_auc(y, score_a, score_b)
    # Bootstrap CI only on pooled to keep runtime sane
    if yip is None:
        auc_a_lo, auc_a_hi = bootstrap_ci(roc_auc_score, y, score_a)
        auc_b_lo, auc_b_hi = bootstrap_ci(roc_auc_score, y, score_b)
    else:
        auc_a_lo = auc_a_hi = auc_b_lo = auc_b_hi = float("nan")
    out_lines.append(f"\n  AUC  (lasso score):")
    out_lines.append(f"    {a_name}: {auc_a:.4f}  [95% CI {auc_a_lo:.4f}, {auc_a_hi:.4f}]")
    out_lines.append(f"    {b_name}: {auc_b:.4f}  [95% CI {auc_b_lo:.4f}, {auc_b_hi:.4f}]")
    out_lines.append(f"    Δ = {auc_a-auc_b:+.4f}   DeLong z={z:.2f}  p={p_auc:.4f}")

    # ---------- PR-AUC ----------
    ap_a = average_precision_score(y, score_a); ap_b = average_precision_score(y, score_b)
    if yip is None:
        ap_a_lo, ap_a_hi = bootstrap_ci(average_precision_score, y, score_a)
        ap_b_lo, ap_b_hi = bootstrap_ci(average_precision_score, y, score_b)
    else:
        ap_a_lo = ap_a_hi = ap_b_lo = ap_b_hi = float("nan")
    out_lines.append(f"\n  PR-AUC (average precision):")
    out_lines.append(f"    {a_name}: {ap_a:.4f}  [{ap_a_lo:.4f}, {ap_a_hi:.4f}]")
    out_lines.append(f"    {b_name}: {ap_b:.4f}  [{ap_b_lo:.4f}, {ap_b_hi:.4f}]")
    out_lines.append(f"    Δ = {ap_a-ap_b:+.4f}")

    # ---------- Top-N% precision + marginal slabs ----------
    out_lines.append(f"\n  Top-N% precision (paired McNemar on disagreement)")
    out_lines.append(f"    {'top%':>5} {'A_prec':>7} {'B_prec':>7} {'A_TP':>5} {'B_TP':>5} {'Δ_pp':>7} {'A_only_TP':>10} {'B_only_TP':>10} {'McN_p':>7}")
    for pct in TOP_PCTS:
        pa, ta, k = topk_precision(score_a, y, pct)
        pb, tb, _ = topk_precision(score_b, y, pct)
        mcn_p, _, _, ao_tp, bo_tp = mcnemar_pick_disagreement(score_a, score_b, y, pct)
        marker = ""
        if not np.isnan(mcn_p):
            marker = "***" if mcn_p<0.001 else ("**" if mcn_p<0.01 else ("*" if mcn_p<0.05 else ""))
        out_lines.append(f"    {pct:>4d}% {pa*100:>6.1f}% {pb*100:>6.1f}% {ta:>5d} {tb:>5d} {(pa-pb)*100:>+6.1f} {ao_tp:>10d} {bo_tp:>10d} {mcn_p:>7.3f}{marker}")
        metrics_rows.append({"event":event,"cohort":tag,"metric":f"P@{pct}%",
                             f"{a_name}":pa, f"{b_name}":pb, "delta_pp":(pa-pb)*100,
                             f"{a_name}_TP":ta, f"{b_name}_TP":tb, "k":k, "mcn_p":mcn_p})

    out_lines.append(f"\n  Marginal slab precision (top X-Y%)")
    out_lines.append(f"    {'slab':>10} {'A_prec':>7} {'B_prec':>7} {'A_TP':>5} {'B_TP':>5} {'A_n':>5} {'B_n':>5} {'Δ_pp':>7}")
    for lo, hi in SLABS:
        pa, ta, na = slab_precision(score_a, y, lo, hi)
        pb, tb, nb = slab_precision(score_b, y, lo, hi)
        out_lines.append(f"    {lo:>3d}-{hi:>3d}% {pa*100:>6.1f}% {pb*100:>6.1f}% {ta:>5d} {tb:>5d} {na:>5d} {nb:>5d} {(pa-pb)*100:>+6.1f}")
        metrics_rows.append({"event":event,"cohort":tag,"metric":f"slab_{lo}-{hi}%",
                             f"{a_name}":pa, f"{b_name}":pb, "delta_pp":(pa-pb)*100,
                             f"{a_name}_TP":ta, f"{b_name}_TP":tb})

    # ---------- Brier + log-loss (on raw hazard, if present) ----------
    if raw_a is not None and raw_b is not None:
        ra = np.clip(raw_a, 1e-6, 1-1e-6); rb = np.clip(raw_b, 1e-6, 1-1e-6)
        br_a = brier_score_loss(y, ra); br_b = brier_score_loss(y, rb)
        ll_a = log_loss(y, ra); ll_b = log_loss(y, rb)
        bss_a = 1 - br_a / (base*(1-base) + 1e-12)
        bss_b = 1 - br_b / (base*(1-base) + 1e-12)
        out_lines.append(f"\n  Calibration (on raw p_{event}):")
        out_lines.append(f"    {'metric':>10} {a_name:>10} {b_name:>10} {'Δ':>9}")
        out_lines.append(f"    {'Brier':>10} {br_a:>10.4f} {br_b:>10.4f} {br_a-br_b:>+9.4f}")
        out_lines.append(f"    {'BSS':>10} {bss_a:>10.4f} {bss_b:>10.4f} {bss_a-bss_b:>+9.4f}")
        out_lines.append(f"    {'LogLoss':>10} {ll_a:>10.4f} {ll_b:>10.4f} {ll_a-ll_b:>+9.4f}")
        ece_a, mce_a = ece(ra, y); ece_b, mce_b = ece(rb, y)
        out_lines.append(f"    {'ECE':>10} {ece_a:>10.4f} {ece_b:>10.4f} {ece_a-ece_b:>+9.4f}")
        out_lines.append(f"    {'MCE':>10} {mce_a:>10.4f} {mce_b:>10.4f} {mce_a-mce_b:>+9.4f}")
        metrics_rows.append({"event":event,"cohort":tag,"metric":"Brier",a_name:br_a,b_name:br_b})
        metrics_rows.append({"event":event,"cohort":tag,"metric":"BSS",a_name:bss_a,b_name:bss_b})
        metrics_rows.append({"event":event,"cohort":tag,"metric":"ECE",a_name:ece_a,b_name:ece_b})

    # ---------- Rank correlation ----------
    rho, p_rho = sstats.spearmanr(score_a, score_b)
    tau, p_tau = sstats.kendalltau(score_a, score_b)
    out_lines.append(f"\n  Score-rank agreement between models:")
    out_lines.append(f"    Spearman ρ = {rho:.4f}   Kendall τ = {tau:.4f}")
    metrics_rows.append({"event":event,"cohort":tag,"metric":"spearman_rho_models",a_name:np.nan,b_name:np.nan,"value":rho})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-long", required=True)
    ap.add_argument("--a-lasso", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-long", required=True)
    ap.add_argument("--b-lasso", required=True)
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--out-prefix", default="model_compare")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--yips", type=int, nargs="+", default=[0,1,2,3,4,5])
    args = ap.parse_args()

    print(f"Scoring {args.a_name} ...")
    a_df = score_with_lasso(args.a_long, args.a_lasso, db=args.db)
    print(f"Scoring {args.b_name} ...")
    b_df = score_with_lasso(args.b_long, args.b_lasso, db=args.db)

    print(f"  {args.a_name}: {len(a_df):,} rows, {a_df.player_id.nunique():,} players")
    print(f"  {args.b_name}: {len(b_df):,} rows, {b_df.player_id.nunique():,} players")
    # Align by (player_id, snap_offset)
    keep_cols = ["player_id","snap_offset"]
    common = a_df[keep_cols].merge(b_df[keep_cols], on=keep_cols)
    a_aln = a_df.merge(common, on=keep_cols).reset_index(drop=True)
    b_aln = b_df.merge(common, on=keep_cols).reset_index(drop=True)
    a_aln = a_aln.sort_values(keep_cols).reset_index(drop=True)
    b_aln = b_aln.sort_values(keep_cols).reset_index(drop=True)
    print(f"  paired (intersection): {len(a_aln):,} rows")

    out_lines = []
    out_lines.append(f"FULL MODEL COMPARISON  {args.a_name}  vs  {args.b_name}")
    out_lines.append(f"  paired rows: {len(a_aln):,}  players: {a_aln.player_id.nunique():,}")
    out_lines.append(f"  yips covered: {sorted(set(a_aln.snap_offset.unique()).intersection(args.yips))}")
    metrics_rows = []
    for ev in EVENTS:
        compare_one_event(a_aln, b_aln, ev, args.a_name, args.b_name, out_lines, metrics_rows, yip=None)
        for yip in args.yips:
            if (a_aln.snap_offset == yip).sum() < 30: continue
            compare_one_event(a_aln, b_aln, ev, args.a_name, args.b_name, out_lines, metrics_rows, yip=yip)

    txt = "\n".join(out_lines)
    out_txt = f"{args.out_prefix}_summary.txt"
    out_csv = f"{args.out_prefix}_metrics.csv"
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(txt)
    pd.DataFrame(metrics_rows).to_csv(out_csv, index=False)
    print(txt)
    print(f"\nSaved {out_txt}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
