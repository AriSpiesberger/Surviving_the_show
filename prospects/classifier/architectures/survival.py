"""
prospects/classifier/architectures/survival.py
==================================

Discrete-time hazard survival model for prospect career events.

Architecture (per event E):
    h_E(t, X) = P(event E triggers in year t  |
                  event E not triggered by start of year t,
                  features X observed through year t-1)

Each h_E is a binary HistGradientBoosting classifier. The eligibility filter
("event not triggered yet") naturally handles censoring — a player who never
established gets a row for every year of their career labeled 0, and a player
who established in 2017 contributes rows up through 2017 (label 1 for 2017,
0 for 2014/2015/2016).

Training corpus: every player drafted in our window. NO filter on having
MiLB stats — a player with empty features still provides a "no transition"
training row that informs the hazard at year t given pedigree alone.

Cumulative P(event by horizon T) = 1 - product_{t=current+1..T} (1 - h_E(t, X_t))

At inference we age the feature vector by one year per step (years_in_pro,
age_at_as_of advance; stats stay frozen at the latest observed snapshot).

Output artifact:
    {event_int: {"hazard": fitted_HGB,
                 "feature_names": list[str]}, ...}

Usage:
    python -m prospects.classifier.architectures.survival \\
        [--db prospects_snapshot.db] \\
        [--max-draft-year 2020] \\
        [--horizon 15] \\
        [--out models/event_classifiers_v1.0_survival.pkl]
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


class _PlattCalibrator:
    """Two-parameter Platt sigmoid: P_cal = 1 / (1 + exp(-(a * raw + b))).

    Fit via logistic regression with raw_p as the single feature. Smooth,
    monotone, continuous — no step artifacts like isotonic on small N.

    Pickle note: instances may have been serialized under `__main__` if
    survival.py was run with `python -m`. We register the class into
    whatever `__main__` is at import time (see below) so existing pickle
    files load cleanly from any other script.
    """
    def __init__(self):
        self._lr: LogisticRegression | None = None
        # Stored for serialization sanity
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_PlattCalibrator":
        x = np.asarray(raw, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y, dtype=np.int8)
        # Guard against all-same-class
        if y.sum() == 0 or y.sum() == len(y):
            self._lr = None
            return self
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        self._lr.fit(x, y)
        self.a = float(self._lr.coef_.ravel()[0])
        self.b = float(self._lr.intercept_.ravel()[0])
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=np.float64).reshape(-1, 1)
        if self._lr is None:
            return np.clip(x.ravel(), 0.0, 1.0)
        return self._lr.predict_proba(x)[:, 1]


class _BetaCalibrator:
    """Three-parameter beta calibration (Kull et al. 2017).

        P_cal(p) = sigmoid(a * log(p) + b * log(1 - p) + c)

    This is a strict superset of Platt (Platt = constraint a = -b). The
    extra degree of freedom lets the calibration curve have different
    slopes at low vs high raw probabilities, which is exactly the
    asymmetry the v1.11 reliability tables exposed (under-predicting
    mid-tier MLB_DEBUT and over-predicting top-tier).

    Fit as logistic regression on [log(p), log(1 - p)] with intercept.
    Same fit/predict surface as _PlattCalibrator so it's a drop-in.
    """
    EPS = 1e-9

    def __init__(self):
        self._lr: LogisticRegression | None = None
        # Stored for inspection / pickle stability
        self.a: float = 1.0
        self.b: float = -1.0
        self.c: float = 0.0
        # Empirical [min, max] of raw seen at fit time — used to clip
        # at inference so we never extrapolate beyond fitted range.
        self._raw_min: float = 0.0
        self._raw_max: float = 1.0

    @staticmethod
    def _features(raw: np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(raw, dtype=np.float64),
                    _BetaCalibrator.EPS, 1.0 - _BetaCalibrator.EPS)
        return np.column_stack([np.log(r), np.log(1.0 - r)])

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_BetaCalibrator":
        raw_a = np.asarray(raw, dtype=np.float64).ravel()
        y_a = np.asarray(y, dtype=np.int8).ravel()
        if y_a.sum() == 0 or y_a.sum() == len(y_a):
            self._lr = None
            return self
        X = self._features(raw_a)
        # C=1.0 regularizes toward the Platt symmetric case (a ≈ -b).
        # The unregularized fit (C=1e6) blew up at extreme raw values
        # for rare events with sparse val data — the intercept became a
        # free parameter that produced absurd predictions at raw values
        # far from the cluster of training samples.
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=400)
        self._lr.fit(X, y_a)
        coef = self._lr.coef_.ravel()
        self.a = float(coef[0])
        self.b = float(coef[1])
        self.c = float(self._lr.intercept_.ravel()[0])
        # Store the 1st / 99th percentile of the raw distribution as the
        # effective fitted range. Predicting outside this range was
        # producing wild extrapolations; we now clip.
        self._raw_min = float(np.quantile(raw_a, 0.01))
        self._raw_max = float(np.quantile(raw_a, 0.99))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        if self._lr is None:
            return np.clip(np.asarray(raw, dtype=np.float64).ravel(),
                           0.0, 1.0)
        # Clip to the empirical raw range we saw at fit time. Beta's
        # log(p) + log(1-p) parameterization extrapolates aggressively
        # outside the training support — a 2026 elite prospect can have
        # raw P(STAR) far higher than anything in the historical val
        # cohort, and uncapped beta would map that to ~1.0. Capping at
        # _raw_max keeps inference consistent with what we calibrated.
        r = np.clip(np.asarray(raw, dtype=np.float64).ravel(),
                    self._raw_min, self._raw_max)
        X = self._features(r)
        return self._lr.predict_proba(X)[:, 1]


# Pickle-compat shim: if any other script loads a model whose calibrator
# was pickled under "__main__", expose this class there so pickle resolves.
import sys as _sys
_main = _sys.modules.get("__main__")
if _main is not None:
    if not hasattr(_main, "_PlattCalibrator"):
        setattr(_main, "_PlattCalibrator", _PlattCalibrator)
    if not hasattr(_main, "_BetaCalibrator"):
        setattr(_main, "_BetaCalibrator", _BetaCalibrator)

from prospects.features.scouting import (
    FEATURE_NAMES,
    N_FEATURES,
    build_scouting_features,
    load_baselines,
)


# ---- Baselines: loaded once on first use ----
_BASELINES_CACHE: dict | None = None
_BASELINES_PATH_DEFAULT = "baselines/milb_baselines.json"


def _baselines() -> dict:
    """Lazy-load + cache the MiLB league baselines. Survival callers don't
    need to pass them through explicitly — same surface as windowed.py."""
    global _BASELINES_CACHE
    if _BASELINES_CACHE is None:
        _BASELINES_CACHE = load_baselines(_BASELINES_PATH_DEFAULT)
    return _BASELINES_CACHE


def build_windowed_features(prospect, stats, as_of_year, milb_only=True):
    """Shim so callers keep the windowed.py signature. Delegates to the
    full scouting feature builder using cached baselines."""
    return build_scouting_features(
        prospect, stats, as_of_year, _baselines(), milb_only=milb_only,
    )
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


MODEL_VERSION = "v1.0-survival"

EVENT_TRIGGER_COL = {
    # v1.11: TOP_100_PROSPECT added as a trainable hazard. Trigger year =
    # first year the player appeared on the BBC top-100 (populated by
    # the outcomes rebuild from prospect_rankings).
    CareerEvent.TOP_100_PROSPECT: "year_top_100",
    CareerEvent.MLB_DEBUT: "mlb_debut_year",
    CareerEvent.ESTABLISHED_MLB: "year_established_mlb",
    CareerEvent.ALL_STAR_ONCE: "year_all_star_once",
    CareerEvent.ALL_STAR_THREE_PLUS: "year_all_star_three",
    CareerEvent.MAJOR_AWARD: "year_major_award",
    # HOF_TRAJECTORY intentionally dropped — not a target we model. Removing
    # it here makes fit_landmark_hazards skip the head (it only fits events
    # present in this map) and excludes HOF from the STAR/ELITE pools below.
}

# Key under which the exit/dropout hazard is stored in the hazards dict.
EXIT_KEY = "_EXIT_BASEBALL"

# Pooled "elite tier" event: any of AS3+ or MAJOR_AWARD.
# Trigger year = min of whichever components fired. (HOF dropped — not a
# target we care about; it only ever overlapped already-elite players.)
ELITE_KEY = "_ELITE"
ELITE_COMPONENT_COLS = (
    "year_all_star_three", "year_major_award",
)

# Pooled "star tier" event: ANY major-league recognition. Union of
# All-Star Once and the ELITE components. Used in v1.5+ to merge AS1
# and ELITE into a single stable rare-event prediction. The previous
# AS1/ELITE split produced unstable predictions on a tiny positive
# count; merging gives the hazard model more training signal and
# eliminates the structural inversion (raw_ELITE > raw_AS1) seen for
# specific players (Hao-Yu Lee, etc.).
STAR_KEY = "_STAR"
STAR_COMPONENT_COLS = (
    "year_all_star_once",
    "year_all_star_three",
    "year_major_award",
)

# Last year for which we have outcomes resolved.
MAX_OBS_YEAR = 2025


def _last_active_year(player: dict, stats_by_pid: dict) -> int | None:
    """Year the player was last observed playing organized baseball.
    Uses career_outcomes.final_mlb_year if MLB-reached, else max season_year
    in their season_stats. Returns None if no signal."""
    fy = player.get("final_mlb_year")
    pid = player.get("player_id")
    rows = stats_by_pid.get(pid, [])
    stat_max = max((s.get("season_year") for s in rows
                    if s.get("season_year") is not None), default=None)
    if fy is None and stat_max is None:
        return None
    if fy is None:
        return int(stat_max)
    if stat_max is None:
        return int(fy)
    return int(max(fy, stat_max))


def _trigger_year(player_row: dict, event) -> int | None:
    """Earliest trigger year for the event. `event` is a CareerEvent or the
    ELITE_KEY string (pooled across AS3+/MAJOR_AWARD)."""
    if event == ELITE_KEY or event == STAR_KEY:
        cols = (STAR_COMPONENT_COLS if event == STAR_KEY
                else ELITE_COMPONENT_COLS)
        candidates: list[int] = []
        for col in cols:
            v = player_row.get(col)
            if v is None:
                continue
            try:
                candidates.append(int(v))
            except (ValueError, TypeError):
                continue
        return min(candidates) if candidates else None
    col = EVENT_TRIGGER_COL[event]
    v = player_row.get(col)
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def build_hazard_panel(
    db: ProspectDB,
    max_draft_year: int = 2020,
    min_year: int = 2005,
    max_year: int = MAX_OBS_YEAR,
    include_ifa: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], list[int], list[dict]]:
    """For every player in cohort (drafted + optional IFAs), emit one row
    per year of their career (start_year+1 .. max_year).

    Start-year semantics:
        - Drafted players: draft_year (must be <= max_draft_year).
        - IFA players: first observed season_stats year.

    Features built MiLB-only with as_of = year-1 (so we never peek at the
    label year).
    """
    with db._connect() as conn:
        prospects = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory,
                   o.events_json, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (p.draft_year IS NOT NULL AND p.draft_year <= ?)
               OR (? = 1 AND COALESCE(p.is_international, 0) = 1)
        """, (max_draft_year, 1 if include_ifa else 0)).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        # Top-100 prospect rankings (Baseball America etc.). Loaded once
        # and attached to each prospect dict as a list of (year, rank).
        # Feature builder computes as-of-year-aware aggregates so we
        # never leak future rankings into a snapshot prediction.
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source FROM prospect_rankings"
            ).fetchall()
        except Exception:
            rank_rows = []

    stats_by_pid: dict[str, list[dict]] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    rankings_by_pid: dict[str, list[tuple[int, int, str]]] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for p in prospects:
        p["_top100_rankings"] = rankings_by_pid.get(p["player_id"], [])

    if verbose:
        n_draft = sum(1 for p in prospects if p.get("draft_year") is not None)
        n_ifa = len(prospects) - n_draft
        print(f"[panel] {len(prospects):,} prospects "
              f"(drafted {n_draft:,} + IFA {n_ifa:,})")

    # Two-pass build: first count how many panel rows we need, then write
    # directly into a single preallocated float32 array. Avoids holding
    # 500k+ float64 vector copies alongside the final stacked matrix at
    # the peak of the np.vstack call (which has been segfaulting under
    # memory pressure with the expanded 213-feature vector).
    pids: list[str] = []
    years: list[int] = []
    joined: list[dict] = []
    n_skipped = 0
    # Plan rows up front (small footprint — just references) so we can
    # preallocate X exactly.
    plan: list[tuple[dict, list, int]] = []
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        dy = p.get("draft_year")
        if dy is None:
            yrs: list[int] = [int(s["season_year"]) for s in stats
                              if s.get("season_year") is not None]
            if not yrs:
                n_skipped += 1
                continue
            start_year = min(yrs)
        else:
            start_year = int(dy)
        for year in range(max(start_year + 1, min_year), max_year + 1):
            plan.append((p, stats, year))
            pids.append(p["player_id"])
            years.append(year)
            joined.append(p)

    n_rows = len(plan)
    X = np.empty((n_rows, N_FEATURES), dtype=np.float32)
    # Chunked build with aggressive gc. Building features for ~500k rows
    # in one tight Python loop accumulates heap fragmentation that has
    # segfaulted Windows. Releasing references and running gc every
    # CHUNK rows keeps the resident set bounded.
    CHUNK = 5000
    for chunk_start in range(0, n_rows, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_rows)
        for i in range(chunk_start, chunk_end):
            p, stats, year = plan[i]
            vec = build_windowed_features(p, stats, year - 1, milb_only=True)
            X[i, :] = vec
        gc.collect()
        if verbose:
            pct = 100.0 * chunk_end / n_rows
            print(f"  [panel] built {chunk_end:,}/{n_rows:,} rows "
                  f"({pct:.0f}%)", flush=True)
    del plan
    gc.collect()
    if verbose:
        print(f"[panel] {X.shape[0]:,} (player, year) rows across "
              f"{len(set(pids)):,} players  "
              f"(skipped {n_skipped:,} prospects with no observed seasons)")
    return X, pids, years, joined


def labels_and_eligibility(
    joined: list[dict],
    years: list[int],
    event: CareerEvent,
    stats_by_pid: dict | None = None,
    right_censor: bool = False,
    min_years_to_fire: int = 0,
    max_obs_year: int = MAX_OBS_YEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """For a given event, compute:
        eligible[i] = True iff event hadn't triggered by start of year[i]
                     (player drafted by year[i]-1 too)
        y[i] = 1 iff event triggered exactly during year[i]

    Right-censoring (when right_censor=True):
      1) Drop rows where yr > last_active_year for players whose event
         never fired (player exited baseball, no longer at risk).
      2) Drop the final-observation-year row for players whose event has
         not yet fired when there has not been enough time for the event
         to plausibly fire from their entry year. Specifically, drop
         rows where yr == max_obs_year AND (yr - start_year) <
         min_years_to_fire AND event has not fired. This prevents recent
         draftees from being treated as confirmed negatives for slow
         events (ESTABLISHED, STAR, ELITE) when they simply haven't had
         time.
    """
    if right_censor and stats_by_pid is None:
        raise ValueError("right_censor=True requires stats_by_pid")
    last_active_cache: dict[str, int | None] = {}
    start_year_cache: dict[str, int | None] = {}
    n = len(years)
    eligible = np.zeros(n, dtype=bool)
    y = np.zeros(n, dtype=np.int8)
    for i, (p, yr) in enumerate(zip(joined, years)):
        # Label year past the data cutoff is half-resolved: recorded positives
        # are counted while same-year negatives simply haven't happened yet.
        # Drop symmetrically (mirrors landmark_event_rows / exit_landmark_rows).
        if yr > max_obs_year:
            eligible[i] = False
            continue
        trig = _trigger_year(p, event)
        if trig is None:
            if right_censor:
                pid = p["player_id"]
                if pid not in last_active_cache:
                    last_active_cache[pid] = _last_active_year(p, stats_by_pid)
                last = last_active_cache[pid]
                if last is not None and yr > last:
                    # Player exited baseball before this year -> censored.
                    eligible[i] = False
                    continue
                # Plausibility censor: still active but not enough time
                # for the slow event to fire from entry.
                if min_years_to_fire > 0 and yr >= max_obs_year:
                    if pid not in start_year_cache:
                        dy = p.get("draft_year")
                        if dy is not None:
                            start_year_cache[pid] = int(dy)
                        else:
                            stat_yrs = [int(s["season_year"])
                                        for s in stats_by_pid.get(pid, [])
                                        if s.get("season_year") is not None]
                            start_year_cache[pid] = (min(stat_yrs)
                                                     if stat_yrs else None)
                    sy = start_year_cache[pid]
                    if sy is not None and (yr - sy) < min_years_to_fire:
                        eligible[i] = False
                        continue
            # Never triggered, still at risk -> eligible, label 0.
            eligible[i] = True
            y[i] = 0
        else:
            if yr < trig:
                eligible[i] = True
                y[i] = 0
            elif yr == trig:
                eligible[i] = True
                y[i] = 1
            else:
                # yr > trig: already happened, drop
                eligible[i] = False
                y[i] = 0
    return eligible, y


def exit_labels(
    joined: list[dict],
    years: list[int],
    stats_by_pid: dict,
    max_year: int = MAX_OBS_YEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Per (player, year), eligibility and label for the EXIT hazard.

    Eligible iff: year <= last_active_year (the player was still in baseball
                  at the start of year).
    Label = 1 iff: year == last_active_year AND last_active_year < max_year
                  (the player has DEFINITELY exited — we'd see them in stats
                  past this year if they hadn't).
    Players whose last_active_year == max_year are censored: their final
    eligible row is labeled 0 (still in baseball as of cutoff).
    """
    n = len(years)
    eligible = np.zeros(n, dtype=bool)
    y = np.zeros(n, dtype=np.int8)
    last_yr_cache: dict[str, int | None] = {}
    for i, (p, yr) in enumerate(zip(joined, years)):
        pid = p["player_id"]
        if pid not in last_yr_cache:
            last_yr_cache[pid] = _last_active_year(p, stats_by_pid)
        last = last_yr_cache[pid]
        if last is None:
            # No activity at all -> player exited at draft+1 (gave up immediately).
            last = (p.get("draft_year") or yr - 1) + 0
        if yr <= last:
            eligible[i] = True
            if yr == last and last < max_year:
                y[i] = 1
    return eligible, y


def fit_hazards(
    X: np.ndarray,
    pids: list[str],
    years: list[int],
    joined: list[dict],
    db_for_stats: ProspectDB,
    seed: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    verbose: bool = True,
    _return_val_players: bool = False,
) -> dict:
    unique_players = sorted(set(pids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_players))
    n_p = len(unique_players)
    n_test = int(round(test_frac * n_p))
    n_val = int(round(val_frac * n_p))
    test_players = set(unique_players[i] for i in perm[:n_test])
    val_players = set(unique_players[i] for i in perm[n_test:n_test + n_val])
    print(f"  player split: train {n_p - n_test - n_val:,} | "
          f"val {n_val:,} | test {n_test:,}")

    split = np.array([
        "test" if p in test_players else ("val" if p in val_players else "train")
        for p in pids
    ])

    print(f"\n{'Event':<22} {'n_tr':>8} {'pos_tr':>7} {'n_te':>8} {'pos_te':>7} "
          f"{'AUC_te':>7} {'Brier_te':>9}")
    print("-" * 80)

    results: dict = {}  # CareerEvent -> dict, plus EXIT_KEY / ELITE_KEY / STAR_KEY
    train_events: list = list(CareerEvent.all_events()) + [ELITE_KEY, STAR_KEY]
    for event in train_events:
        if (event not in (ELITE_KEY, STAR_KEY)
                and event not in EVENT_TRIGGER_COL):
            continue  # TOP_100 / TOP_25 don't have ranking data
        eligible, y_all = labels_and_eligibility(joined, years, event)

        tr = (split == "train") & eligible
        te = (split == "test") & eligible
        X_tr, y_tr = X[tr], y_all[tr]
        X_te, y_te = X[te], y_all[te]

        n_pos_tr = int(y_tr.sum())
        ev_label = event.name if hasattr(event, "name") else str(event)
        if n_pos_tr < 10 or n_pos_tr > X_tr.shape[0] - 10:
            print(f"{ev_label:<22} {X_tr.shape[0]:>8,d} {n_pos_tr:>7,d} "
                  f"{X_te.shape[0]:>8,d} {int(y_te.sum()):>7,d}     n/a       n/a")
            continue

        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            min_samples_leaf=30, random_state=seed,
            early_stopping=True, n_iter_no_change=10,
            validation_fraction=0.1,
        ).fit(X_tr, y_tr)

        if X_te.shape[0] > 0 and int(y_te.sum()) > 0:
            p_te = clf.predict_proba(X_te)[:, 1]
            try:
                auc = roc_auc_score(y_te, p_te)
            except Exception:
                auc = float("nan")
            brier = brier_score_loss(y_te, p_te)
            print(f"{ev_label:<22} {X_tr.shape[0]:>8,d} {n_pos_tr:>7,d} "
                  f"{X_te.shape[0]:>8,d} {int(y_te.sum()):>7,d} "
                  f"{auc:>7.3f} {brier:>9.4f}")
        else:
            p_te = np.empty(0, dtype=np.float64)
            print(f"{ev_label:<22} {X_tr.shape[0]:>8,d} {n_pos_tr:>7,d} "
                  f"{X_te.shape[0]:>8,d} {int(y_te.sum()):>7,d} "
                  f"  (test_frac=0; CV gives honest test metrics)")

        results[event] = {
            "hazard": clf,
            "feature_names": list(FEATURE_NAMES),
        }
        # Drop per-event arrays before next iteration
        del X_tr, y_tr, X_te, y_te, p_te, eligible, y_all
        gc.collect()

    # --- Exit / dropout hazard ---
    # Need stats_by_pid for the eligibility computation; pull it here.
    with db_for_stats._connect() as conn:
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    elig_e, y_e_all = exit_labels(joined, years, stats_by_pid)
    tr_e = (split == "train") & elig_e
    te_e = (split == "test") & elig_e
    X_tr_e, y_tr_e = X[tr_e], y_e_all[tr_e]
    X_te_e, y_te_e = X[te_e], y_e_all[te_e]
    n_pos_e = int(y_tr_e.sum())
    if n_pos_e >= 10:
        clf_e = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=30, random_state=seed,
        ).fit(X_tr_e, y_tr_e)
        if X_te_e.shape[0] > 0 and int(y_te_e.sum()) > 0:
            p_te_e = clf_e.predict_proba(X_te_e)[:, 1]
            try:
                auc_e = roc_auc_score(y_te_e, p_te_e)
            except Exception:
                auc_e = float("nan")
            brier_e = brier_score_loss(y_te_e, p_te_e)
            print(f"{'EXIT_BASEBALL':<22} {X_tr_e.shape[0]:>8,d} {n_pos_e:>7,d} "
                  f"{X_te_e.shape[0]:>8,d} {int(y_te_e.sum()):>7,d} "
                  f"{auc_e:>7.3f} {brier_e:>9.4f}")
        else:
            print(f"{'EXIT_BASEBALL':<22} {X_tr_e.shape[0]:>8,d} {n_pos_e:>7,d} "
                  f"{X_te_e.shape[0]:>8,d} {int(y_te_e.sum()):>7,d} "
                  f"  (test_frac=0)")
        results[EXIT_KEY] = {
            "hazard": clf_e,
            "feature_names": list(FEATURE_NAMES),
        }
    return results


def predict_cumulative(
    hazards: dict[CareerEvent, dict],
    prospect: dict,
    season_stats: list[dict],
    current_year: int,
    horizon: int = 15,
) -> dict[CareerEvent, float]:
    """Simulate the next `horizon` years and return P(event by then) per event.

    Features are aged each step (years_in_pro and age advance); stat features
    stay frozen at the latest observed pre-current_year snapshot. The
    accumulated probability is 1 - product over the horizon of (1 - hazard).
    """
    yip_idx = FEATURE_NAMES.index("years_in_pro")
    age_idx = FEATURE_NAMES.index("age_at_as_of")

    # Per-event "already triggered" check
    out: dict[CareerEvent, float] = {}
    for event in CareerEvent.all_events():
        if event not in hazards:
            out[event] = 0.0
            continue
        trig = _trigger_year(prospect, event)
        if trig is not None and trig <= current_year:
            # Already happened
            out[event] = 1.0
            continue
        clf = hazards[event]["hazard"]
        surv = 1.0
        for step in range(horizon):
            year = current_year + 1 + step
            X = build_windowed_features(
                prospect, season_stats, year - 1, milb_only=True,
            ).reshape(1, -1)
            # The build_windowed_features already advances years_in_pro and
            # age based on draft_year and same-year stats; no extra aging needed.
            h = float(clf.predict_proba(X)[0, 1])
            surv *= (1.0 - h)
        out[event] = 1.0 - surv
    return out


def predict_cumulative_batch(
    hazards: dict[CareerEvent, dict],
    prospects: list[dict],
    stats_by_pid: dict,
    current_year: int,
    horizon: int = 15,
) -> dict[CareerEvent, np.ndarray]:
    """Vectorized batch survival simulator.

    Key behavior: features are built ONCE at `current_year` (using all
    observed stats through current_year). For each subsequent horizon step,
    only the time-varying scalars (`years_in_pro`, `age_at_as_of`,
    `years_in_current_system`) are advanced. Performance/level features
    stay frozen at the latest observed snapshot. This matches reality at
    inference: we don't know future stats; we know the player's current
    snapshot and how they'd age.
    """
    event_keys = [k for k in hazards if k != EXIT_KEY]
    n = len(prospects)
    surv = {e: np.ones(n, dtype=np.float64) for e in event_keys}
    triggered = {e: np.zeros(n, dtype=bool) for e in event_keys}
    eligible = {e: np.ones(n, dtype=bool) for e in event_keys}
    # Mean / SD of time-to-event, conditional on event triggering in horizon.
    # Accumulate sum_p, sum_t*p, sum_t^2*p across horizon steps.
    sum_p = {e: np.zeros(n, dtype=np.float64) for e in event_keys}
    sum_tp = {e: np.zeros(n, dtype=np.float64) for e in event_keys}
    sum_t2p = {e: np.zeros(n, dtype=np.float64) for e in event_keys}
    for i, p in enumerate(prospects):
        for e in event_keys:
            trig = _trigger_year(p, e)
            if trig is not None and trig <= current_year:
                eligible[e][i] = False
                triggered[e][i] = True

    stats_lists = [stats_by_pid.get(p["player_id"], []) for p in prospects]
    X0 = np.vstack([
        build_windowed_features(prospects[i], stats_lists[i],
                                current_year, milb_only=True)
        for i in range(n)
    ])

    yip_i = FEATURE_NAMES.index("years_in_pro")
    age_i = FEATURE_NAMES.index("age_at_as_of")
    yics_i = FEATURE_NAMES.index("years_in_current_system")

    # P(still in baseball at start of year t). Cumulative.
    in_baseball = np.ones(n, dtype=np.float64)
    has_exit = EXIT_KEY in hazards
    exit_clf = hazards[EXIT_KEY]["hazard"] if has_exit else None

    # Prerequisite chain. Downstream events accumulate timing weighted by
    # P(prereq has happened by start of step), so E[t] respects the event
    # ladder by construction. Unconditional cumP outputs are unchanged
    # (calibrators stay valid).
    from prospects.schema import CareerEvent as _CE
    prereq_map = {
        _CE.ESTABLISHED_MLB:       _CE.MLB_DEBUT,
        _CE.ALL_STAR_ONCE:         _CE.ESTABLISHED_MLB,
        _CE.ALL_STAR_THREE_PLUS:   _CE.ALL_STAR_ONCE,
        _CE.MAJOR_AWARD:           _CE.ESTABLISHED_MLB,
        _CE.HOF_TRAJECTORY:        _CE.ESTABLISHED_MLB,
        ELITE_KEY:                 _CE.ESTABLISHED_MLB,
        STAR_KEY:                  _CE.ESTABLISHED_MLB,
    }

    def _prereq_cumP_at_step(e):
        """cumP of prereq at the START of this step = 1 - surv[prereq]."""
        prereq = prereq_map.get(e)
        if prereq is None or prereq not in surv:
            return np.ones(n, dtype=np.float64)
        return 1.0 - surv[prereq]

    # Cumulative P(event happened by year t) = 1 - prod over s<=t of
    # P(no event in s | still in baseball at start of s).
    for step in range(horizon):
        X = X0.copy()
        col_yip = X[:, yip_i]
        col_age = X[:, age_i]
        col_yics = X[:, yics_i]
        mask_yip = ~np.isnan(col_yip)
        mask_age = ~np.isnan(col_age)
        mask_yics = ~np.isnan(col_yics)
        X[mask_yip, yip_i] = col_yip[mask_yip] + (step + 1)
        X[mask_age, age_i] = col_age[mask_age] + (step + 1)
        X[mask_yics, yics_i] = col_yics[mask_yics] + (step + 1)

        t_step = step + 1  # year 1, 2, 3 ... from now
        # Snapshot prereq cumP at start of step BEFORE updating surv. This
        # is P(prereq has occurred by year t) used to weight timing
        # contributions for downstream events at year t.
        prereq_cumP_now = {e: _prereq_cumP_at_step(e) for e in event_keys}

        for e in event_keys:
            mask = eligible[e]
            if mask.sum() == 0:
                continue
            h = hazards[e]["hazard"].predict_proba(X[mask])[:, 1]
            # Unconditional step_p drives the surv update + cumP output
            # (so the trained calibrators still apply).
            step_p = surv[e][mask] * in_baseball[mask] * h
            # Conditioned step_p drives the timing expectation: weight by
            # P(prereq has happened by start of this step). For DEBUT, the
            # weight is 1 (no prereq).
            step_p_t = step_p * prereq_cumP_now[e][mask]
            sum_p[e][mask] += step_p_t
            sum_tp[e][mask] += t_step * step_p_t
            sum_t2p[e][mask] += (t_step ** 2) * step_p_t
            surv[e][mask] *= (1.0 - in_baseball[mask] * h)

        # Update in_baseball for next step
        if has_exit:
            h_exit = exit_clf.predict_proba(X)[:, 1]
            in_baseball = in_baseball * (1.0 - h_exit)

    # P(event triggered cumulative). For already-triggered players, =1.
    out: dict = {}
    for e in event_keys:
        p_raw = 1.0 - surv[e]
        p_raw = np.where(triggered[e], 1.0, p_raw)
        cal = hazards[e].get("calibrator")
        if cal is not None:
            p_cal = np.asarray(cal.predict(p_raw), dtype=np.float64)
        else:
            p_cal = p_raw
        # Already-triggered events are realized, not predicted — force to 1.0
        # (Beta calibrator otherwise maps raw=1.0 to ~0.936 for TOP_100.)
        p_cal = np.where(triggered[e], 1.0, p_cal)
        out[e] = p_cal
        out[("raw", e)] = p_raw

        # Conditional E[T] and SD[T] from the accumulated step probabilities.
        # If sum_p is ~0 (player has near-zero P of event), the conditional
        # time is undefined → emit NaN.
        sp = sum_p[e]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_t = np.where(sp > 1e-9, sum_tp[e] / sp, np.nan)
            e_t2 = np.where(sp > 1e-9, sum_t2p[e] / sp, np.nan)
            var_t = np.clip(e_t2 - mean_t ** 2, 0.0, None)
            sd_t = np.sqrt(var_t)
        # Already-triggered players: time is 0 (it's already happened).
        mean_t = np.where(triggered[e], 0.0, mean_t)
        sd_t = np.where(triggered[e], 0.0, sd_t)
        out[("mean_t", e)] = mean_t
        out[("sd_t", e)] = sd_t
    return out


def fit_cumulative_calibrators(
    hazards: dict[CareerEvent, dict],
    db: ProspectDB,
    val_players: set[str],
    max_obs_year: int = MAX_OBS_YEAR,
    horizon: int = 15,
    method: str = "sigmoid",
    verbose: bool = True,
) -> dict:
    """For each event, score VAL players' cumulative P (at draft_year+2
    snapshot or first-observed+2 for IFAs) and fit a calibrator against
    realized {0, 1}.

    Args:
        method: "sigmoid" (Platt, continuous; default) or "iso" (isotonic
                step function — finer empirical fit but produces ties).
    """
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.events_json, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (p.draft_year IS NOT NULL AND p.draft_year <= 2018)
               OR COALESCE(p.is_international, 0) = 1
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source FROM prospect_rankings"
            ).fetchall()
        except Exception:
            rank_rows = []
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    rankings_by_pid: dict[str, list] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for r in rows:
        r["_top100_rankings"] = rankings_by_pid.get(r["player_id"], [])

    val_rows = [r for r in rows if r["player_id"] in val_players]
    if verbose:
        print(f"\n[iso-cal] fitting isotonic per event on {len(val_rows)} val players")

    # Multi-snapshot expansion: each val player is scored at several
    # snapshots (default draft+1, +2, +3, +5). Same player contributes
    # multiple cal samples, each with a distinct feature vector but the same
    # eventual outcome label. ~4x the cal set size.
    from collections import defaultdict
    snapshot_offsets = (1, 2, 3, 5)
    groups_by_year: dict[int, list[dict]] = defaultdict(list)
    for r in val_rows:
        dy = r.get("draft_year")
        if dy is None:
            pid = r["player_id"]
            stat_yrs: list[int] = [
                int(s["season_year"]) for s in stats_by_pid.get(pid, [])
                if s.get("season_year") is not None
            ]
            if not stat_yrs:
                continue
            start = min(stat_yrs)
        else:
            start = int(dy)
        for offset in snapshot_offsets:
            cur_year = start + offset
            if cur_year >= max_obs_year:
                continue
            groups_by_year[cur_year].append(r)

    if verbose:
        sizes = sorted(((y, len(g)) for y, g in groups_by_year.items()))
        print(f"  [iso-cal] {len(groups_by_year)} snapshot-year groups: "
              f"{sizes[:3]}...{sizes[-3:]}")

    # Score every player at their snapshot year, collect per-event preds & labels
    # Keys are either int (CareerEvent enum value) or the ELITE_KEY string.
    score_keys = [e for e in hazards if e != EXIT_KEY]
    def _ev_key(e):
        return e if isinstance(e, str) else int(e)
    per_event_preds: dict = {_ev_key(e): [] for e in score_keys}
    per_event_real: dict = {_ev_key(e): [] for e in score_keys}
    for cur_year, group in groups_by_year.items():
        sub_stats = {r["player_id"]: stats_by_pid.get(r["player_id"], []) for r in group}
        out = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=cur_year, horizon=horizon,
        )
        for i, r in enumerate(group):
            for ev in score_keys:
                k = _ev_key(ev)
                per_event_preds[k].append(float(out[ev][i]))
                trig = _trigger_year(r, ev)
                per_event_real[k].append(
                    int(trig is not None and trig <= max_obs_year)
                )

    isotonics: dict = {}
    for event in hazards:
        if event == EXIT_KEY:
            continue
        k = _ev_key(event)
        preds, real = per_event_preds[k], per_event_real[k]
        preds_a = np.array(preds)
        real_a = np.array(real)
        if real_a.sum() < 3 or real_a.sum() > len(real_a) - 3:
            if verbose:
                print(f"  {event.name:<22} skip (positives={int(real_a.sum())})")
            continue
        if method == "iso":
            calib = IsotonicRegression(out_of_bounds="clip",
                                       y_min=0.0, y_max=1.0)
            calib.fit(preds_a, real_a)
        elif method == "sigmoid":
            calib = _PlattCalibrator().fit(preds_a, real_a)
        elif method == "beta":
            calib = _BetaCalibrator().fit(preds_a, real_a)
        else:
            raise ValueError(f"Unknown calibrator method: {method!r}")
        isotonics[event] = calib
        if verbose:
            top10_idx = np.argsort(preds_a)[::-1][:max(1, len(preds_a) // 10)]
            label = event.name if hasattr(event, "name") else str(event)
            print(f"  {label:<22} n={len(preds_a)} pos={int(real_a.sum())} "
                  f"top10%: pre {preds_a[top10_idx].mean():.3f} -> "
                  f"{method[:3]} {calib.predict(preds_a[top10_idx]).mean():.3f} "
                  f"(real {real_a[top10_idx].mean():.3f})")
    return isotonics


def save_hazards(hazards: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, m in hazards.items():
        key = k if isinstance(k, str) else int(k)
        serializable[key] = m
    with open(path, "wb") as f:
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_hazards(path: str) -> dict:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out: dict = {}
    for k, v in raw.items():
        if isinstance(k, str):
            out[k] = v
        else:
            out[CareerEvent(int(k))] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects_snapshot.db")
    parser.add_argument("--max-draft-year", type=int, default=2020)
    parser.add_argument("--max-year", type=int, default=MAX_OBS_YEAR)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--calibrator", choices=["sigmoid", "iso", "beta"],
                        default="beta",
                        help="Calibration method. 'beta' (default in v1.12+) "
                             "is a 3-param generalization of Platt that "
                             "handles asymmetric miscalibration; 'sigmoid' is "
                             "the v1.x Platt fit; 'iso' is isotonic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.075,
                        help="Fraction of players reserved for calibrator fit "
                             "(default 0.075 = 7.5%%)")
    parser.add_argument("--test-frac", type=float, default=0.0,
                        help="Fraction reserved for test (default 0; use the "
                             "5-fold CV results for honest test metrics)")
    parser.add_argument("--panel", default=None,
                        help="Optional path to pre-built panel npz "
                             "(from build_panel.py). Skips panel build.")
    parser.add_argument("--out",
                        default="models/event_classifiers_v1.0_survival.pkl")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    import os
    if args.panel and os.path.exists(args.panel):
        print(f"Loading prebuilt panel from {args.panel}")
        with np.load(args.panel, allow_pickle=True) as d:
            X = d["X"].astype(np.float32, copy=False)
            pids = d["pids"].tolist()
            years = d["years"].tolist()
        joined_path = args.panel.replace(".npz", ".joined.pkl")
        with open(joined_path, "rb") as fh:
            joined = pickle.load(fh)
    else:
        X, pids, years, joined = build_hazard_panel(
            db, max_draft_year=args.max_draft_year, max_year=args.max_year,
        )
    # Reproduce the player split: test (--test-frac), val (--val-frac),
    # rest goes to train.
    unique_players = sorted(set(pids))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(unique_players))
    n_p = len(unique_players)
    n_test = int(round(args.test_frac * n_p))
    n_val = int(round(args.val_frac * n_p))
    val_players = set(unique_players[i] for i in perm[n_test:n_test + n_val])
    print(f"Player split: train {n_p - n_test - n_val:,} | "
          f"val {n_val:,} | test {n_test:,}  "
          f"(--val-frac={args.val_frac}, --test-frac={args.test_frac})")

    hazards = fit_hazards(X, pids, years, joined, db, seed=args.seed,
                          val_frac=args.val_frac, test_frac=args.test_frac)

    # Per-event isotonic on cumulative P using val players' draft+2 snapshots
    cals = fit_cumulative_calibrators(
        hazards, db, val_players,
        max_obs_year=args.max_year, horizon=args.horizon,
        method=args.calibrator,
    )
    for e, c in cals.items():
        hazards[e]["calibrator"] = c

    save_hazards(hazards, args.out)
    print(f"\nSaved hazards + calibrators -> {args.out}  (version: {MODEL_VERSION})")


if __name__ == "__main__":
    main()
