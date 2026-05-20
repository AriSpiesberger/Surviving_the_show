# 5-fold Cross-validation Evaluation Plan

## Goal

Produce out-of-fold (OOF) predictions for **every player in the cohort** so we evaluate on 100% of the data, then run a complete statistical test suite to know what the model actually does — with confidence intervals everywhere we make a claim.

---

## Phase 1: Generate OOF predictions

### 1.1 — Per-fold training

For seed `s` in `[0, 1, 2, 3, 4]`:

1. Split the player universe into 5 stratified-by-bucket groups using `seed=s`. Player-grouped (all rows of a player land in the same fold).
2. Fold `f`: train on the other 4 folds (~80% of data), hold out fold `f` for OOF prediction.
3. Train all hazards on the training 4-fold subset.
4. From the training subset, carve a calibration slice (~10% of training, NOT touching the held-out fold) and fit Beta calibrators.
5. Predict on the held-out fold at its snapshot year.
6. Store predictions tagged with fold index.

**Output**: `oof_predictions_v1.13.csv` — one row per player, columns include:
- player_id, name, fold, snap_year
- `p_<EVENT>` (calibrated), `p_<EVENT>_raw` for each of MLB_DEBUT, ESTABLISHED, AS1, STAR, TOP_100
- `realized_<EVENT>`, `eligible_at_snap_<EVENT>`, `trigger_year_<EVENT>`
- Bucket label (R1 / R2-R3 / R4-R10 / R10+ / IFA)

### 1.2 — Data integrity checks (gate before running tests)

| Check | Pass criterion |
|---|---|
| Every player predicted exactly once | `count_distinct(player_id) == count(rows)` |
| Every player has a fold assignment | `fold IS NOT NULL` for all rows |
| Predicted probabilities in [0, 1] | `min(p) >= 0` and `max(p) <= 1` |
| No NaN predictions | `count_null(p) == 0` |
| Cohort coverage matches | n_oof = n_test_cohort (eligible after dedup, debut filter) |
| Fold balance | `std(fold_sizes) / mean(fold_sizes) < 0.05` |
| Per-fold calibrators converged | a > 0, b < 0, c finite for all (fold, event) |

---

## Phase 2: Aggregate metrics (with confidence intervals)

For each event, on the full OOF prediction set:

### 2.1 — Ranking quality

| Metric | Definition | CI method |
|---|---|---|
| AUC | ROC-AUC | DeLong's variance (parametric) and bootstrap (2000 resamples) — agreement check |
| AUC vs 0.5 | Test "model adds ranking value" | One-sided 95% CI on AUC; fail if CI lower bound ≤ 0.5 |
| Brier score | mean((p − y)²) | Bootstrap CI |
| Brier-skill | 1 − Brier / Brier_base, base = predicting mean(y) everywhere | Bootstrap CI; flag positive iff lower CI bound > 0 |
| Log-loss skill | 1 − LL / LL_base | Bootstrap CI |

### 2.2 — Calibration

| Metric | Definition | Test |
|---|---|---|
| ECE (Expected Calibration Error) | sum_bin w_bin · \|p_mean − y_mean\| | Bootstrap CI |
| Spiegelhalter Z-test | Standardized residual of (sum p − sum y) under perfect calibration | Two-sided p-value; flag if p < 0.05 (calibration significantly off) |
| Hosmer–Lemeshow Ĥ | Chi-squared on 10 deciles of predicted prob | p-value; flag if p < 0.05 |
| Per-bin reliability | n, predicted_mean, realized_rate, Wilson 95% CI per bin | "in_band" flag = predicted_mean inside CI |

### 2.3 — Discrimination at the top

| Metric | Definition | CI |
|---|---|---|
| Precision @ top-N% | (true positives in top-k predictions) / k for k=ceil(N% · cohort) | Wilson CI on precision |
| Recall @ top-N% | tp / total_positives | Wilson CI |
| Lift @ top-N% | precision / base_rate | Bootstrap CI on the ratio |

Compute at N% in {1, 5, 10, 20}.

---

## Phase 3: Per-bucket metrics (with CIs)

Repeat all of Phase 2's tests per `(event × draft_bucket)` cell.

Bucket = {R1, R2-R3, R4-R10, R10+, IFA}.

### 3.1 — Per-cell tests

| Cell test | Definition | Threshold |
|---|---|---|
| AUC significantly > 0.5 | bootstrap 95% lower CI on AUC | > 0.5 |
| Brier-skill significantly > 0 | bootstrap 95% lower CI on Brier-skill | > 0 |
| Calibration not rejected | Spiegelhalter p-value | > 0.05 |
| Top-1%/5%/20% lift > 1 | bootstrap 95% lower CI on lift | > 1 |

### 3.2 — Statistical power

For each cell, compute the **minimum detectable AUC improvement** at 80% power given the cell's positives count. Cells with insufficient power get flagged as "underpowered — cannot statistically distinguish skill from chance."

Formula: MDE on AUC ≈ √(0.25 / n_positives) for the 80% power, alpha=0.05 case. Cells with MDE > 0.10 are "underpowered."

---

## Phase 4: Stability checks

### 4.1 — Cross-fold variance

For each metric in each (event, bucket) cell, compute std-dev across 5 folds. Flag any cell where `std / mean > 0.30` as "fold-unstable" — the metric isn't robust to which 20% of data we hold out.

### 4.2 — In-fold vs OOF gap

Train one model on 100% of data, score on the same data, compare to OOF predictions:
- ΔAUC = AUC(in-fold) − AUC(OOF)
- If ΔAUC > 0.05 for an event, the model is overfitting that event.

### 4.3 — Train/test base rate consistency

For each fold, check that the base rate of positives in train vs test differs by less than 30%. Mismatched base rates mean a non-representative split.

---

## Phase 5: Honest assessment summary

A single decision-grade table per event:

| | Aggregate AUC | AUC CI | Calibration p | ECE | Skill score | Underpowered cells | Fold-unstable cells |
|---|---|---|---|---|---|---|---|
| MLB_DEBUT | ? | ? | ? | ? | ? | ? | ? |
| ESTABLISHED | ? | ? | ? | ? | ? | ? | ? |
| AS1 | ? | ? | ? | ? | ? | ? | ? |
| STAR | ? | ? | ? | ? | ? | ? | ? |
| TOP_100 | ? | ? | ? | ? | ? | ? | ? |

Per (event, bucket) cell heatmap with three indicators:
- 🟢 AUC + Brier-skill + lift all CI-significantly positive
- 🟡 At least one significantly positive, others not
- 🔴 No metric significantly distinguishable from baseline
- ⚪ Underpowered (insufficient positives to test)

---

## Deliverables

| File | Contents |
|---|---|
| `oof_predictions_v1.13.csv` | One row per player with all event predictions, fold, snap, realized |
| `cv_metrics_aggregate_v1.13.csv` | Per-event aggregate metrics with bootstrap CIs |
| `cv_metrics_per_bucket_v1.13.csv` | Per (event, bucket) metrics with bootstrap CIs |
| `cv_calibration_v1.13.csv` | Per-event reliability bins + p-values |
| `cv_topn_v1.13.csv` | Per (event, bucket) top-N% precision/recall/lift with CIs |
| `cv_stability_v1.13.csv` | Per-cell cross-fold variance |
| `cv_assessment_v1.13.txt` | Plain-English summary + decision-grade table |

---

## Implementation order

1. **Phase 1.1**: Write a CV runner — most of the work. Trains 5 models, refits calibrators per fold, predicts OOF. ~30 min runtime estimate.
2. **Phase 1.2**: Integrity gate. If anything fails, stop and fix before running stats.
3. **Phase 2 + 3 + 4**: Statistical tests on the OOF CSV. Bounded compute (bootstrap is the expensive bit; 2000 resamples × 25 cells × 5 events ≈ minutes).
4. **Phase 5**: Generate the assessment report.

---

## Open decisions before we start

1. **Bootstrap resamples**: 2000 is standard; 5000 for tighter CIs at the rare-event extremes if compute allows.
2. **Per-fold calibration vs global**: I propose per-fold (each model has its own calibrator, fit on its own training-internal val slice). This is correct for an honest evaluation but adds complexity. Alternative: pool predictions and fit one global calibrator on a held-out 10%. Per-fold is the right call.
3. **Stratification key**: stratify folds by draft bucket × debut-outcome (binary), so each fold has a representative sample of every bucket. Without stratification, folds can be 5pp different on R1 base rates and metrics get noisy.
4. **Time-of-prediction snapshot**: keep current (draft_year + 2 for drafted, first_milb_year + 2 for IFAs). Should we also evaluate at a later snapshot to test how predictions evolve? Probably not in this run — that's a separate stability question.

Want me to lock these decisions and start Phase 1?
