"""Regenerate evaluation/README.md from the CSVs in evaluation/v2.0b_landmark/.

Static prose is templated here; every TABLE is rebuilt from the latest
per_bucket / per_yip / per_level / thresholds CSVs + headline.json, so the
README never drifts from the numbers.

    python -m scripts_v17.validate.gen_eval_readme
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "evaluation" / "v2.0b_landmark"
OUT = REPO / "evaluation" / "README.md"

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
BUCKET_ORDER = ["ALL", "R1", "R2-R3", "R4-R10", "R10+", "IFA"]
LEVEL_ORDER = ["ALL", "RK", "A-", "A", "A+", "AA", "AAA", "NONE"]
COLS = "| {grp} | {n} | {pos} | {base:.2f}% | {auc} | {ap} | {lift} | {sp} | {prec} | {rec} | {f1} | {tp} | {fp} | {fn} |"
HDR = ("| {g} | n | pos | base% | AUC | AP | AP_lift | spearman | precision "
       "| recall | F1 | TP | FP | FN |\n"
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def _row(r, grp):
    empty = int(r["pos"]) == 0
    f3 = lambda v: "—" if (empty or pd.isna(v)) else f"{v:.3f}"
    return COLS.format(
        grp=grp, n=int(r["n"]), pos=int(r["pos"]), base=r["base_rate"] * 100,
        auc=f3(r["auc"]), ap=f3(r["ap"]),
        lift="—" if empty or pd.isna(r["ap_lift"]) else f"{r['ap_lift']:.1f}×",
        sp=f3(r["spearman_rho"]),
        prec=f"{r['precision']:.3f}" if int(r["predicted_positives"]) > 0 else "—",
        rec="—" if empty else f"{r['recall']:.3f}",
        f1="—" if (empty or pd.isna(r["f1"]) or r["f1"] == 0) else f"{r['f1']:.3f}",
        tp=int(r["tp"]), fp=int(r["fp"]), fn=int(r["fn"]))


def _section(df, group_col, order, label):
    out = []
    for ev in EVENTS:
        sub = df[df.event == ev]
        rows = []
        keys = order if order else sorted(sub[group_col].unique())
        for k in keys:
            cell = sub[sub[group_col] == k]
            if not cell.empty:
                rows.append(_row(cell.iloc[0], str(k)))
        if rows:
            out.append(f"\n#### {ev}\n\n{HDR.format(g=label)}\n" + "\n".join(rows))
    return "\n".join(out)


def main():
    bucket = pd.read_csv(EV / "per_bucket_validation.csv")
    yip = pd.read_csv(EV / "per_yip_validation.csv")
    level = pd.read_csv(EV / "per_level_validation.csv")
    head = json.loads((EV / "headline.json").read_text())
    thr = pd.read_csv(EV / "MLB_DEBUT_thresholds_at_p60.csv")

    # headline (ALL bucket per event) + weighted
    hl = ["| Event | n | base% | AP | lift | AUC | spearman | precision | recall | F1 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ev in EVENTS:
        r = bucket[(bucket.event == ev) & (bucket.bucket == "ALL")].iloc[0]
        prec = f"{r['precision']:.3f}" if int(r["predicted_positives"]) > 0 else "—"
        f1 = "—" if (pd.isna(r["f1"]) or r["f1"] == 0) else f"{r['f1']:.3f}"
        hl.append(f"| {ev} | {int(r['n'])} | {r['base_rate']*100:.2f}% | "
                  f"**{r['ap']:.3f}** | {r['ap_lift']:.1f}× | {r['auc']:.3f} | "
                  f"{r['spearman_rho']:.3f} | {prec} | {r['recall']:.3f} | {f1} |")
    hl.append(f"| **weighted-AP** | | | **{head['weighted_ap']:.3f}** | | | | | | |")

    thr_rows = ["| yip | threshold | n_above | TP | precision | recall | n_total | n_pos_total |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in thr.iterrows():
        t = "—" if pd.isna(r["threshold"]) else f"{r['threshold']:.3f}"
        pr = "—" if pd.isna(r["precision"]) else f"{r['precision']:.3f}"
        rc = "—" if pd.isna(r["recall"]) else f"{r['recall']:.3f}"
        thr_rows.append(f"| {int(r['yip'])} | {t} | {int(r['n_above'])} | "
                        f"{int(r['tp_above'])} | {pr} | {rc} | "
                        f"{int(r['n_total'])} | {int(r['n_pos_total'])} |")

    md = f"""# Held-out validation — v2.0b landmark (+ scouting grades)

Reproducible evaluation of the v2.0b landmark stack against the **10% val
player slice** of the v1.17 seed=42 split — players neither the landmark
hazards nor the joint XGBoost head trained on. Validation universe: drafted
players with `draft_year ≤ 2020` (plus IFAs), realized window through 2026.

**This revision** adds point-in-time scouting features from the FanGraphs
Board (2017–2026) + Trouble-With-The-Curve (2013–2019): 76 grade/physical/
velo/spin/rank/ETA/risk columns in the hazard panel (no-lookahead, season ≤
snapshot), plus a compact 5-column current-snapshot summary
(`scout_fv, scout_ovr_rank, scout_eta_gap, scout_risk, scout_is_scouted`) fed
directly into the joint XGB. HOF_TRAJECTORY was dropped from the event set.

## Stack

| Layer | Model | Trained on |
|---|---|---|
| Hazards (per-fold OOF) | `scratch/v20b_oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 folds (val pids excluded). HistGBT, **default HP**, 314 features (incl. 76 scouting). |
| XGB head | `models/joint_xgb_v2.0b_oof.pkl` | OOF stacked CSV. Default HP via `fit_joint_xgb_v2.py` — `max_depth=6, lr=0.05, early_stop=25, best_iter=196`. FEAT = hazard probs + age/yip + scouting summary. Honest at XGB layer. |

Scouting features are point-in-time (latest grade with `season ≤ snapshot`),
so val features never see the future. ~4% of val rows are scouted (only ranked
prospects get grades); the rest are NaN/sentinel, which HistGBT handles.

## Headline (ALL bucket, threshold = 0.60)

{chr(10).join(hl)}

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters.)

## Per-bucket (threshold = 0.60)
{_section(bucket, "bucket", BUCKET_ORDER, "bucket")}

## Per-yip (threshold = 0.60)
{_section(yip, "snap_offset", list(range(11)), "yip")}

## Per-level (threshold = 0.60)
{_section(level, "cur_level", LEVEL_ORDER, "level")}

## Threshold @ precision ≥ 0.60 (MLB_DEBUT per yip)

{chr(10).join(thr_rows)}

## Statistics glossary

| Metric | Meaning |
|---|---|
| `ap` | Average Precision = AU-PR. Headline rare-event metric. |
| `ap_lift` | `ap / base_rate` — how many × random the ranking is. |
| `auc` | Area under ROC. Insensitive to class imbalance. |
| `spearman_rho` | Rank correlation between score and realized 0/1. |
| `precision/recall/f1` | At threshold 0.60. `—` = undefined (no predicted positives / no positives). |
| `bucket` | Draft pedigree: R1, R2-R3, R4-R10, R10+ (rounds 11+), IFA. |
| `snap_offset` (yip) | Years since entry. |
| `cur_level` | Player's level at snapshot: RK/A-/A/A+/AA/AAA/NONE. |

## Reproducing

```bash
# data: scouting grades (FG board scrape + TWTC) -> point-in-time table
python -m scripts.scrape_fangraphs_board --start 2017 --end 2026   # needs curl_cffi
python -m scripts.build_fg_crosswalk
python -m scripts.build_scouting_grades

# model: OOF folds + hazards + joint XGB (grades flow through both layers)
python -m scripts_v17.train.run_v2_0b_oof
python -m scripts_v17.train.fit_joint_xgb_v2 \\
    --fit results/training/v2.0b_oof_stacked_long.csv \\
    --val results/training/v2.0b_oof_val_long.csv \\
    --out models/joint_xgb_v2.0b_oof.pkl

# validation tables -> evaluation/v2.0b_landmark/, then this README
python -m scripts_v17.validate.validate_full --long results/training/v2.0b_oof_val_long.csv \\
    --xgb-model models/joint_xgb_v2.0b_oof.pkl \\
    --time-to-debut-model models/time_to_debut_v1.18_prod.pkl --target-precision 0.60 --out-prefix v20b_clean
python -m scripts_v17.validate.regen_eval_v2_0b_honest --threshold 0.60
python -m scripts_v17.validate.regen_full_eval_v2_0b
python -m scripts_v17.validate.gen_eval_readme
```
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT} ({len(md):,} chars)")
    print(f"weighted-AP = {head['weighted_ap']:.4f}")


if __name__ == "__main__":
    main()
