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

    md = f"""# Held-out validation — v2.0b landmark (censoring-corrected)

Reproducible evaluation of the v2.0b landmark stack against the **10% val
player slice** of the v1.17 seed=42 split — players neither the landmark
hazards nor the joint XGBoost head trained on. Validation universe: drafted
players with `draft_year ≤ 2020` (plus IFAs).

**Yardstick: RESOLVED outcomes only.** Raw labels are right-censored (an event
is recorded only if it occurred by the data cutoff), so a model that predicts
*eventual* outcome is unfairly penalized for events that will happen after the
cutoff. The joint XGB therefore trains on — and is evaluated on — **resolved
rows**: the event was observed, OR the player had ≥6 forward years without it.
This fixed a severe debut undercount (AAA arms read ~2% debut before; realistic
now). Knobs: `--censor-window 6` (`fit_joint_xgb_v2.py`), `--resolved-window 6`
(regen scripts). The **hazards** are survival models — censoring-aware by
construction — and need no filter.

**Data integrity (this revision):** birthdates backfilled for 2024–25 draft
classes, FG/TWTC crosswalk 89%→96%, trade-aware `current_org` (latest-season
affiliate → parent org via MLB Stats API), IFA entry-year anchors, signing-bonus
backfill. Point-in-time scouting (FanGraphs Board 2017–26 + Trouble-With-The-
Curve 2013–19): 76 grade/physical/velo/rank/ETA columns in the hazard panel
(no-lookahead, season ≤ snapshot) + a 5-col current-snapshot summary
(`scout_fv, scout_ovr_rank, scout_eta_gap, scout_risk, scout_is_scouted`) fed to
the XGB. HOF_TRAJECTORY dropped from the event set.

## Stack

| Layer | Model | Trained on |
|---|---|---|
| Hazards (per-fold OOF, eval) | `scratch/v20b_oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 (val pids excluded). HistGBT, default HP, 314 features (incl. 76 scouting). Survival → censoring-aware. |
| Hazards (production) | `models/event_classifiers_v2.0b_prod.pkl` | 100% of ≤2020 data. Scores the 2026 cohort (entry 2024–26 — not in training, so no leakage). |
| XGB head | `models/joint_xgb_v2.0b_{{oof,prod}}.pkl` | OOF stacked CSV, **censoring-corrected (`--censor-window 6`)**. Default HP (`max_depth=6, lr=0.05`). FEAT = hazard probs + age/yip + 5-col scouting summary (19). Honest at XGB layer. |

Scouting features are point-in-time (latest grade with `season ≤ snapshot`),
so features never see the future. ~4% of rows are scouted (only ranked
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
# data integrity backfills (MLB Stats API) + scouting grades
python -m prospects.ingestion.backfills.birthdate_backfill      # 2024-25 DOB
python -m prospects.ingestion.backfills.org_backfill            # trade-aware current_org
python -m scripts.scrape_fangraphs_board --start 2017 --end 2026   # needs curl_cffi
python -m scripts.build_fg_crosswalk      # 96% match (needs DOB backfill first)
python -m scripts.build_scouting_grades

# model: OOF folds + hazards + censoring-corrected joint XGB (--censor-window 6
# is wired into run_v2_0b_oof stage 6 and train_v2_0b_prod stage 1)
python -m scripts_v17.train.run_v2_0b_oof
python -m scripts_v17.train.train_v2_0b_prod    # 100% prod hazards + W=6 XGB + score 2026

# validation on the RESOLVED yardstick (positives + >=6 fwd-yr negatives)
python -c "import pandas as pd; v=pd.read_csv('results/training/v2.0b_oof_val_long.csv'); \\
  r=[c for c in v if c.startswith('realized_')]; \\
  v[(v.years_fwd>=6)|(v[r].sum(1)>0)].to_csv('results/training/v2.0b_oof_val_long_resolved.csv',index=False)"
python -m scripts_v17.validate.regen_eval_v2_0b_honest \\
    --val-long results/training/v2.0b_oof_val_long_resolved.csv --threshold 0.60
python -m scripts_v17.validate.regen_full_eval_v2_0b \\
    --val-long results/training/v2.0b_oof_val_long_resolved.csv
python -m scripts_v17.validate.gen_eval_readme
```
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT} ({len(md):,} chars)")
    print(f"weighted-AP = {head['weighted_ap']:.4f}")


if __name__ == "__main__":
    main()
