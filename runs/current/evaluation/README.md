# Held-out validation — v2.4 (raw-feature bag + recent-cohort augmentation)

Reproducible evaluation of the v2.4 stack against the **15% val player
slice** (`model/train/make_split`, seed=42) — players neither the landmark
hazards nor the joint XGBoost head trained on. Validation universe: drafted
players with `draft_year ≤ 2020` **and international signees whose first
non-MLB season is ≤ 2020** (IFAs entered the split on 2026-09-10; before that
the joint layer trained on ~14k of them but val held none, so they were never
held-out-evaluated).

The published sheet is scored by v3.5, which blends this stack with further components; its metrics are in [`../evaluation_v3`](../evaluation_v3).

## Headline (ALL bucket, h=6, threshold = 0.60)

| Event | n | base% | AP | lift | AUC | spearman | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 35144 | 1.10% | **0.487** | 44.2× | 0.969 | 0.170 | 0.812 | 0.202 | 0.323 |
| MLB_DEBUT | 35392 | 8.53% | **0.591** | 6.9× | 0.930 | 0.416 | 0.774 | 0.253 | 0.382 |
| ESTABLISHED_MLB | 35392 | 2.72% | **0.363** | 13.3× | 0.936 | 0.246 | 0.743 | 0.027 | 0.052 |
| STAR_PLUS_ELITE | 35392 | 0.43% | **0.139** | 32.2× | 0.949 | 0.102 | — | 0.000 | — |
| **weighted-AP** | | | **0.434** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Details

- [Evaluation notes and corrections](notes.md)
- [Stack](stack.md)
- [Reliability](reliability.md)
- [Per-horizon trajectory](trajectory.md)
- [Per-bucket, per-yip and per-level metrics](slices.md)
- [Glossary and reproduction](reference.md)
