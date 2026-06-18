"""Landmark-trained discrete-time hazard model for prospect career events.

DIFF VS contemporaneous survival.py:
  - The contemporaneous architecture emits ONE training row per (player, year)
    with features as_of (year-1) and label "event fires in year". At inference,
    survival.predict_cumulative_batch walks the same model forward k=1..15
    steps by aging only yip/age/yics on a frozen feature vector. That LOCF
    aging is OOD vs training for k>=1: training never saw frozen-features +
    advanced-yip rows, so the model under-projects fast risers and over-
    projects stallers, with the bias concentrated on slow events (STAR/ELITE)
    where most cumulative mass accrues at high k.

  - This module fixes the train/inference mismatch by emitting one row per
    (player, landmark S, k) triplet:
        features at as_of = landmark S
        label = "event fires exactly in year S+k"
        k is an explicit feature column (FEATURE_NAMES + ["horizon_offset_k"])
    Then inference uses the same model with k set to the current step, and
    yip/age/yics held frozen at S. Training and inference draw from the same
    distribution by construction.

  - Trade: we pay K_event-fold more rows per landmark, and late-k slow-event
    labels are heavily right-censored (small effective N at the tail). Bias
    goes down, variance at high k goes up. That is the better trade for a
    probability we calibrate downstream.

The contemporaneous architecture in survival.py stays as the v1.18 baseline.
This module trains v1.18b.

Module structure (mirror of survival.py to keep the swap point small):
  K_PER_EVENT           : max forward offset emitted per event
  LANDMARK_K_FEATURE    : the name of the appended k column
  FEATURE_NAMES_LM      : base FEATURE_NAMES + [LANDMARK_K_FEATURE]
  build_landmark_panel  : two-pass build of (X_lm, landmark_index, joined_lm)
  landmark_event_rows   : per-event filter to (triplet_idx, y, k) using
                          existing eligibility + right-censoring rules,
                          applied at year = S+k
  exit_landmark_rows    : same for the EXIT classifier
  fit_landmark_hazards  : per-event HistGBT fit; output dict matches the
                          survival.py shape so downstream loaders work
  predict_cumulative_batch_landmark : k-as-feature version of the inference
                          loop; preserves competing-risk composition, exit
                          hazard step, prereq-weighted timing math.
"""
from __future__ import annotations

import gc
import sys
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from prospects.classifier.architectures.survival import (
    ELITE_COMPONENT_COLS, ELITE_KEY, EVENT_TRIGGER_COL, EXIT_KEY, MAX_OBS_YEAR,
    STAR_COMPONENT_COLS, STAR_KEY,
    _last_active_year, _trigger_year, build_windowed_features,
)
from prospects.features.partial_sample import partial_for_features
from prospects.features.scouting import FEATURE_NAMES, N_FEATURES
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


MODEL_VERSION = "v1.18b-landmark"

# Maximum forward offset emitted per event during training. Tuned so we keep
# meaningful mass per event while not wasting compute on labels that are
# almost-always right-censored at the recent landmarks. Slow events (STAR/
# ELITE) cap at the same K as fast ones because their tail past k=10 is
# dominated by censoring noise anyway — the gain comes from cleanly fitting
# k=5..10, not from chasing k=15.
K_PER_EVENT: dict = {
    CareerEvent.TOP_100_PROSPECT:     10,
    CareerEvent.MLB_DEBUT:            10,
    CareerEvent.ESTABLISHED_MLB:      12,
    CareerEvent.ALL_STAR_ONCE:        12,
    CareerEvent.ALL_STAR_THREE_PLUS:  10,
    CareerEvent.MAJOR_AWARD:          10,
    ELITE_KEY:                        10,
    STAR_KEY:                         10,
}
K_EXIT = 10  # for the EXIT_BASEBALL hazard

# Right-censoring policy mirror of the contemporaneous EVENT_POLICY in
# train_full_v14d.py. Applied at the LABEL year (S+k), not at the landmark.
EVENT_POLICY_LM: dict = {
    "TOP_100_PROSPECT":     (True, 0),
    "MLB_DEBUT":            (True, 0),
    "ESTABLISHED_MLB":      (True, 4),
    "ALL_STAR_ONCE":        (True, 4),
    "ALL_STAR_THREE_PLUS":  (True, 6),
    "MAJOR_AWARD":          (True, 5),
    "HOF_TRAJECTORY":       (True, 10),
    "ELITE":                (True, 5),
    "STAR":                 (True, 4),
}

LANDMARK_K_FEATURE = "horizon_offset_k"
# The augmented feature schema: base 238 features + k. INFERENCE writes k
# into the last column. TRAINING uses this same layout.
FEATURE_NAMES_LM: list[str] = list(FEATURE_NAMES) + [LANDMARK_K_FEATURE]
N_FEATURES_LM: int = N_FEATURES + 1
K_FEATURE_INDEX: int = N_FEATURES  # zero-indexed position of the k column


# ---------- helpers ----------

def _ename(e) -> str:
    return e.name if hasattr(e, "name") else str(e).lstrip("_")


def _start_year(player: dict, stats_by_pid: dict) -> int | None:
    """Player start year: draft_year for drafted, earliest non-MLB
    season_year for IFAs. Used for the min_years_to_fire censor."""
    dy = player.get("draft_year")
    if dy is not None and int(player.get("is_international") or 0) == 0:
        return int(dy)
    stat_yrs = [int(s["season_year"])
                for s in stats_by_pid.get(player["player_id"], [])
                if s.get("season_year") is not None
                and (s.get("level") or "").upper() != "MLB"]
    if stat_yrs:
        return int(min(stat_yrs))
    if dy is not None:
        return int(dy)
    return None


# ---------- panel ----------

def build_landmark_panel(
    db: ProspectDB,
    max_draft_year: int = 2020,
    min_landmark_year: int = 2007,
    max_landmark_year: int | None = None,
    include_ifa: bool = True,
    verbose: bool = True,
    partial_seed: int | None = None,
) -> tuple[np.ndarray, list[str], list[int], list[dict], dict]:
    """Build the per-landmark feature matrix.

    partial_seed : int | None
        Training-time partial-season augmentation. When None (default) the
        panel is built from complete seasons exactly as before. When set, the
        current (landmark-year S) stint of each landmark is stochastically
        down-sampled to an in-progress partial line via
        partial_for_features, so the hazards train on the real mid-season
        feature manifold instead of only complete seasons. Deterministic in
        (player_id, S, partial_seed).

    Returns
    -------
    X_lm : float32 (n_landmark, N_FEATURES)
        Feature matrix at the LANDMARK granularity (one row per (player,
        landmark)). The k column is NOT included here — it gets stamped
        on per-event during row expansion.
    pids : list[str], length n_landmark
        Player id per landmark row.
    landmark_years : list[int], length n_landmark
        Landmark year S per row.
    joined : list[dict], length n_landmark
        The prospect dict per landmark row (carries trigger_year info).
    stats_by_pid : dict
        Same season_stats lookup the contemporaneous build emits. Required
        by the per-event row expander for right-censoring.

    The K-fold row expansion happens inside landmark_event_rows. We keep
    the landmark-granularity matrix because storing 6M+ float32 rows ×
    239 cols (~5.5GB) would OOM on typical workstations; per-event
    expansion lets each fit work over only that event's eligible
    triplets.
    """
    if max_landmark_year is None:
        max_landmark_year = MAX_OBS_YEAR - 1  # need at least k=1 observable

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
        print(f"[landmark-panel] {len(prospects):,} prospects "
              f"(drafted {n_draft:,} + IFA {n_ifa:,})  "
              f"landmark range {min_landmark_year}..{max_landmark_year}")

    plan: list[tuple[dict, list, int]] = []
    pids: list[str] = []
    landmark_years: list[int] = []
    joined: list[dict] = []
    n_skipped = 0
    n_ifa_capped = 0
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        sy = _start_year(p, stats_by_pid)
        if sy is None:
            n_skipped += 1
            continue
        # C3 (v2.1): cap IFA entry year the SAME way draft year is capped in the
        # SQL. IFAs have no draft_year, so the query's `draft_year <= max` never
        # filtered them — post-cutoff IFAs were leaking into training (and into
        # the "held-out" 2021 walk-forward). Drop IFAs whose entry > cutoff.
        if p.get("draft_year") is None and sy > max_draft_year:
            n_ifa_capped += 1
            continue
        # Landmark range: from start_year+1 (so we have at least one
        # observed pre-landmark season feeding the features) up to
        # max_landmark_year.
        lo = max(sy + 1, min_landmark_year)
        hi = max_landmark_year
        if lo > hi:
            continue
        for S in range(lo, hi + 1):
            plan.append((p, stats, S))
            pids.append(p["player_id"])
            landmark_years.append(S)
            joined.append(p)

    n_rows = len(plan)
    if verbose:
        print(f"[landmark-panel] {n_rows:,} landmark rows planned "
              f"({n_skipped} skipped no start_year, "
              f"{n_ifa_capped:,} IFAs capped at entry<={max_draft_year})")

    X_lm = np.empty((n_rows, N_FEATURES), dtype=np.float32)
    CHUNK = 5000
    for chunk_start in range(0, n_rows, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_rows)
        for i in range(chunk_start, chunk_end):
            p, stats, S = plan[i]
            # Features as-of S. The contemporaneous panel passes year-1
            # so it never peeks at the label year; here the label year is
            # S+k (k>=1), so features at as_of=S are equivalent in
            # information leakage to as_of=year-1 in survival.py.
            # Optionally down-sample season S to a partial in-progress line.
            stats_S = partial_for_features(stats, S, p["player_id"], partial_seed)
            vec = build_windowed_features(p, stats_S, S, milb_only=True)
            X_lm[i, :] = vec
        gc.collect()
        if verbose:
            pct = 100.0 * chunk_end / n_rows
            print(f"  [landmark-panel] built {chunk_end:,}/{n_rows:,} "
                  f"({pct:.0f}%)", flush=True)

    del plan
    gc.collect()
    return X_lm, pids, landmark_years, joined, stats_by_pid


# ---------- per-event row expansion ----------

def landmark_event_rows(
    joined: list[dict],
    landmark_years: list[int],
    event,
    K: int,
    stats_by_pid: dict,
    right_censor: bool = True,
    min_years_to_fire: int = 0,
    max_obs_year: int = MAX_OBS_YEAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand the landmark index to (landmark_idx, k, y) for one event.

    For each (player, landmark S) row in joined, walk k = 1..K and emit a
    training triplet when the (event, year S+k) cell is eligible:

      - If event already fired by year S (trig <= S): emit NOTHING. The
        landmark snapshot would never be scored against this event at
        inference, so it's not informative training material.
      - If event fires at year S+k (trig == S+k): eligible=True, y=1.
      - If event has not fired by year S+k start (trig is None or
        trig > S+k): eligible=True, y=0, subject to right-censoring.
      - If trig is between (S, S+k) i.e. trig < S+k: not eligible (already
        fired before this label year).

    Right-censoring at the LABEL year S+k:
      1) Drop rows for never-firers whose last_active < S+k (exited).
      2) Drop rows where S+k >= max_obs_year AND (S+k - start_year)
         < min_years_to_fire — the event hasn't had time to fire yet.

    Returns
    -------
    landmark_idx : int32 (n_event,)
        Index into landmark_years/joined for each emitted triplet.
    k_arr : int16 (n_event,)
        Horizon offset for each triplet, in {1..K}.
    y : int8 (n_event,)
        Binary label, 1 iff event fires exactly at S+k.
    """
    last_active_cache: dict[str, int | None] = {}
    start_year_cache: dict[str, int | None] = {}
    landmark_idx: list[int] = []
    k_arr: list[int] = []
    y_list: list[int] = []
    for i, (p, S) in enumerate(zip(joined, landmark_years)):
        trig = _trigger_year(p, event)
        # Already fired before landmark: skip the whole player-landmark.
        if trig is not None and trig <= S:
            continue
        pid = p["player_id"]
        # Pre-cache last_active and start_year once per player.
        if right_censor and pid not in last_active_cache:
            last_active_cache[pid] = _last_active_year(p, stats_by_pid)
        if right_censor and pid not in start_year_cache:
            start_year_cache[pid] = _start_year(p, stats_by_pid)
        last = last_active_cache.get(pid) if right_censor else None
        sy = start_year_cache.get(pid) if right_censor else None

        for k in range(1, K + 1):
            label_year = S + k
            # Symmetric censoring: the label year must be fully observed.
            # Past the data cutoff a positive is counted only because it's
            # already recorded, while a same-year negative simply hasn't
            # happened yet (e.g. an August debut at a June refresh) — the C1
            # half-resolved pattern. Drop the whole tail; label_year only grows
            # with k, so break (mirrors exit_landmark_rows). This also drops the
            # year's recorded positives — the price of symmetric censoring.
            if label_year > max_obs_year:
                break
            # Never fires by end of observation: y=0 if at-risk, dropped
            # if censored.
            if trig is None:
                if right_censor:
                    if last is not None and label_year > last:
                        # Player exited before this label year.
                        continue
                    if (min_years_to_fire > 0
                            and label_year >= max_obs_year
                            and sy is not None
                            and (label_year - sy) < min_years_to_fire):
                        # Not enough time for the slow event to fire.
                        continue
                landmark_idx.append(i)
                k_arr.append(k)
                y_list.append(0)
            else:
                # trig is in the FUTURE of landmark (we filtered earlier).
                if trig == label_year:
                    landmark_idx.append(i)
                    k_arr.append(k)
                    y_list.append(1)
                elif trig > label_year:
                    landmark_idx.append(i)
                    k_arr.append(k)
                    y_list.append(0)
                else:
                    # trig < label_year: already fired earlier than k.
                    # Not eligible — drop and stop emitting beyond this
                    # k for this player-landmark.
                    break
    return (np.asarray(landmark_idx, dtype=np.int32),
            np.asarray(k_arr, dtype=np.int16),
            np.asarray(y_list, dtype=np.int8))


def exit_landmark_rows(
    joined: list[dict],
    landmark_years: list[int],
    stats_by_pid: dict,
    K: int = K_EXIT,
    max_obs_year: int = MAX_OBS_YEAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per (player, landmark, k) eligibility + label for the EXIT hazard.

    Exit fires in year Y when last_active_year(player) == Y and the
    player never reached MLB before that year (the contemporaneous
    exit_labels rule).  We follow the same semantics, evaluated at the
    label year S+k.

    Returns landmark_idx, k_arr, y as in landmark_event_rows.
    """
    last_active_cache: dict[str, int | None] = {}
    landmark_idx: list[int] = []
    k_arr: list[int] = []
    y_list: list[int] = []
    for i, (p, S) in enumerate(zip(joined, landmark_years)):
        pid = p["player_id"]
        if pid not in last_active_cache:
            last_active_cache[pid] = _last_active_year(p, stats_by_pid)
        last = last_active_cache[pid]
        debut = p.get("mlb_debut_year")
        # Once a player debuts, they're no longer an "exit candidate" in
        # the prospects sense.
        # Per landmark, walk k=1..K:
        for k in range(1, K + 1):
            label_year = S + k
            if label_year > max_obs_year:
                # Can't observe whether they exit in the future.
                continue
            if last is None:
                # Insufficient data to define last_active.
                continue
            if debut is not None and debut <= label_year:
                # Already in MLB by the label year — not an exit candidate.
                continue
            if label_year < last:
                # Still active in subsequent years.
                landmark_idx.append(i); k_arr.append(k); y_list.append(0)
            elif label_year == last:
                # This is the exit year.
                landmark_idx.append(i); k_arr.append(k); y_list.append(1)
            else:
                # Already exited before label_year — not a training row
                # (they're out of the at-risk set).
                break
    return (np.asarray(landmark_idx, dtype=np.int32),
            np.asarray(k_arr, dtype=np.int16),
            np.asarray(y_list, dtype=np.int8))


# ---------- training ----------

def _assemble_event_X(X_lm: np.ndarray, landmark_idx: np.ndarray,
                      k_arr: np.ndarray) -> np.ndarray:
    """Stack (N_event, N_FEATURES+1). Stream-builds to avoid two copies."""
    n_event = landmark_idx.size
    X_event = np.empty((n_event, N_FEATURES_LM), dtype=np.float32)
    # Bulk gather features. fancy-indexing makes one copy; we then write
    # k into the appended column.
    X_event[:, :N_FEATURES] = X_lm[landmark_idx]
    X_event[:, K_FEATURE_INDEX] = k_arr.astype(np.float32)
    return X_event


_HAZARD_HP_DEFAULTS = dict(
    max_iter=200, max_depth=6, learning_rate=0.05,
    min_samples_leaf=30, early_stopping=True,
    n_iter_no_change=10, validation_fraction=0.1,
)


def _train_event(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 42,
                 hp: dict | None = None,
                 ) -> HistGradientBoostingClassifier:
    """Fit one event's HistGBT. `hp` overrides any defaults — used by
    the Optuna hazards tuner."""
    params = dict(_HAZARD_HP_DEFAULTS)
    if hp:
        params.update(hp)
    return HistGradientBoostingClassifier(
        **params, random_state=seed,
    ).fit(X_tr, y_tr)


def fit_landmark_hazards(
    X_lm: np.ndarray,
    joined: list[dict],
    landmark_years: list[int],
    stats_by_pid: dict,
    train_mask: np.ndarray | None = None,
    seed: int = 42,
    k_per_event: dict | None = None,
    max_obs_year: int = MAX_OBS_YEAR,
    verbose: bool = True,
    hazard_hp: dict | None = None,
) -> dict:
    """Per-event HistGBT fit. Output dict matches survival.fit_hazards
    so load_hazards / save_hazards / downstream Beta calibration code
    work unchanged. Each entry carries the augmented FEATURE_NAMES_LM
    so the inference loop knows there's a k column.

    train_mask : bool array (n_landmark,)
        Per-landmark mask of which rows are in the training slice. For
        v1.18b-prod we'll pass all-True (train on 100%); for v1.18b-test
        the prod pipeline uses an 80/10/10 split on the LANDMARK rows.

    Returns dict keyed by event:
      {event: {"hazard": HistGBT, "feature_names": FEATURE_NAMES_LM,
               "kind": "landmark", "k_max": K_event, "n_train": int,
               "n_pos_train": int}}
    """
    k_per_event = dict(k_per_event or K_PER_EVENT)
    if train_mask is None:
        train_mask = np.ones(X_lm.shape[0], dtype=bool)

    hazards: dict = {}
    train_events = list(CareerEvent.all_events()) + [ELITE_KEY, STAR_KEY]

    if verbose:
        print(f"{'Event':<24} {'policy':<20} {'K':>3} "
              f"{'n_train':>10} {'pos':>7} {'pos_rate':>9}")
        print("-" * 80)

    for event in train_events:
        if (event not in (ELITE_KEY, STAR_KEY)
                and event not in EVENT_TRIGGER_COL):
            continue
        ename = _ename(event)
        rc, min_yrs = EVENT_POLICY_LM.get(ename, (True, 0))
        K = k_per_event.get(event, 10)
        landmark_idx, k_arr, y_all = landmark_event_rows(
            joined, landmark_years, event, K, stats_by_pid,
            right_censor=rc, min_years_to_fire=min_yrs,
            max_obs_year=max_obs_year,
        )
        # Apply train_mask via the landmark index.
        keep = train_mask[landmark_idx]
        landmark_idx = landmark_idx[keep]
        k_arr = k_arr[keep]
        y_all = y_all[keep]
        n = int(y_all.size)
        n_pos = int(y_all.sum())
        if n_pos < 10 or n_pos > n - 10:
            if verbose:
                print(f"{ename:<24} {f'rc={rc},min={min_yrs}':<20} "
                      f"{K:>3} {n:>10,d} {n_pos:>7d}    skip")
            continue
        X_tr = _assemble_event_X(X_lm, landmark_idx, k_arr)
        clf = _train_event(X_tr, y_all, seed=seed, hp=hazard_hp)
        if verbose:
            print(f"{ename:<24} {f'rc={rc},min={min_yrs}':<20} "
                  f"{K:>3} {n:>10,d} {n_pos:>7d} "
                  f"{100.0*n_pos/n:>8.2f}%")
        hazards[event] = {
            "hazard": clf,
            "feature_names": list(FEATURE_NAMES_LM),
            "kind": "landmark",
            "k_max": K,
            "n_train": n,
            "n_pos_train": n_pos,
        }
        del X_tr, landmark_idx, k_arr, y_all
        gc.collect()

    # Exit hazard
    landmark_idx, k_arr, y_all = exit_landmark_rows(
        joined, landmark_years, stats_by_pid, K=K_EXIT,
        max_obs_year=max_obs_year,
    )
    keep = train_mask[landmark_idx]
    landmark_idx = landmark_idx[keep]
    k_arr = k_arr[keep]
    y_all = y_all[keep]
    n = int(y_all.size); n_pos = int(y_all.sum())
    if n_pos >= 10 and n_pos < n - 10:
        X_tr = _assemble_event_X(X_lm, landmark_idx, k_arr)
        clf_e = _train_event(X_tr, y_all, seed=seed)
        if verbose:
            print(f"{EXIT_KEY:<24} {'exit-only':<20} {K_EXIT:>3} "
                  f"{n:>10,d} {n_pos:>7d} {100.0*n_pos/n:>8.2f}%")
        hazards[EXIT_KEY] = {
            "hazard": clf_e,
            "feature_names": list(FEATURE_NAMES_LM),
            "kind": "landmark",
            "k_max": K_EXIT,
            "n_train": n,
            "n_pos_train": n_pos,
        }
    return hazards


# ---------- inference ----------

def predict_cumulative_batch_landmark(
    hazards: dict,
    prospects: list[dict],
    stats_by_pid: dict,
    current_year: int,
    horizon: int = 15,
) -> dict:
    """Vectorized batch survival simulator using landmark hazards.

    DIFF from survival.predict_cumulative_batch:
      - At each step we set X[:, K_FEATURE_INDEX] = step + 1 rather than
        advancing yip/age/yics. The HistGBT was trained with k as a
        feature, so this is a direct in-distribution prediction.

    Same competing-risk composition, exit-hazard product, prereq-
    weighted timing. Outputs match the contemporaneous function so
    downstream consumers see the same dict shape (cumP per event,
    optional ("raw", event), mean_t, sd_t).

    If horizon > k_max for an event, we hold k at the model's training
    K_max for steps beyond that — the model has nothing to say about
    later offsets, so feeding the saturated k is the least-bad option.
    Downstream cumulative composition is unaffected by this clamp at
    the same scale.
    """
    event_keys = [k for k in hazards if k != EXIT_KEY]
    n = len(prospects)
    surv = {e: np.ones(n, dtype=np.float64) for e in event_keys}
    triggered = {e: np.zeros(n, dtype=bool) for e in event_keys}
    eligible = {e: np.ones(n, dtype=bool) for e in event_keys}
    sum_p = {e: np.zeros(n, dtype=np.float64) for e in event_keys}
    sum_tp = {e: np.zeros(n, dtype=np.float64) for e in event_keys}
    sum_t2p = {e: np.zeros(n, dtype=np.float64) for e in event_keys}
    step_haz = {e: np.zeros((n, horizon), dtype=np.float64) for e in event_keys}
    for i, p in enumerate(prospects):
        for e in event_keys:
            trig = _trigger_year(p, e)
            if trig is not None and trig <= current_year:
                eligible[e][i] = False
                triggered[e][i] = True

    stats_lists = [stats_by_pid.get(p["player_id"], []) for p in prospects]
    # Features are built ONCE at current_year. The simulator does NOT age
    # yip/age/yics — that's the whole point. k carries the time information.
    X0 = np.empty((n, N_FEATURES_LM), dtype=np.float32)
    for i in range(n):
        X0[i, :N_FEATURES] = build_windowed_features(
            prospects[i], stats_lists[i], current_year, milb_only=True)
    # k column starts at 0; gets set per step.

    # P(still in baseball at start of year t). Cumulative.
    in_baseball = np.ones(n, dtype=np.float64)
    has_exit = EXIT_KEY in hazards
    exit_clf = hazards[EXIT_KEY]["hazard"] if has_exit else None
    k_max_exit = hazards[EXIT_KEY].get("k_max", K_EXIT) if has_exit else 0

    from prospects.schema import CareerEvent as _CE
    prereq_map = {
        _CE.ESTABLISHED_MLB:     _CE.MLB_DEBUT,
        _CE.ALL_STAR_ONCE:       _CE.ESTABLISHED_MLB,
        _CE.ALL_STAR_THREE_PLUS: _CE.ALL_STAR_ONCE,
        _CE.MAJOR_AWARD:         _CE.ESTABLISHED_MLB,
        _CE.HOF_TRAJECTORY:      _CE.ESTABLISHED_MLB,
        ELITE_KEY:               _CE.ESTABLISHED_MLB,
        STAR_KEY:                _CE.ESTABLISHED_MLB,
    }

    def _prereq_cumP_at_step(e):
        prereq = prereq_map.get(e)
        if prereq is None or prereq not in surv:
            return np.ones(n, dtype=np.float64)
        return 1.0 - surv[prereq]

    for step in range(horizon):
        t_step = step + 1  # year 1, 2, 3 ... from now
        prereq_cumP_now = {e: _prereq_cumP_at_step(e) for e in event_keys}

        for e in event_keys:
            mask = eligible[e]
            if mask.sum() == 0:
                continue
            X = X0[mask].copy()
            k_eff = min(t_step, hazards[e].get("k_max", t_step))
            X[:, K_FEATURE_INDEX] = float(k_eff)
            h = hazards[e]["hazard"].predict_proba(X)[:, 1]
            step_haz[e][mask, step] = h   # raw per-year hazard curve
            step_p = surv[e][mask] * in_baseball[mask] * h
            step_p_t = step_p * prereq_cumP_now[e][mask]
            sum_p[e][mask] += step_p_t
            sum_tp[e][mask] += t_step * step_p_t
            sum_t2p[e][mask] += (t_step ** 2) * step_p_t
            surv[e][mask] *= (1.0 - in_baseball[mask] * h)

        if has_exit:
            X = X0.copy()
            k_eff = min(t_step, k_max_exit)
            X[:, K_FEATURE_INDEX] = float(k_eff)
            h_exit = exit_clf.predict_proba(X)[:, 1]
            in_baseball = in_baseball * (1.0 - h_exit)

    out: dict = {}
    for e in event_keys:
        p_raw = 1.0 - surv[e]
        p_raw = np.where(triggered[e], 1.0, p_raw)
        cal = hazards[e].get("calibrator")
        if cal is not None:
            p_cal = np.asarray(cal.predict(p_raw), dtype=np.float64)
        else:
            p_cal = p_raw
        p_cal = np.where(triggered[e], 1.0, p_cal)
        out[e] = p_cal
        out[("raw", e)] = p_raw

        sp = sum_p[e]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_t = np.where(sp > 1e-9, sum_tp[e] / sp, np.nan)
            e_t2 = np.where(sp > 1e-9, sum_t2p[e] / sp, np.nan)
            var_t = np.clip(e_t2 - mean_t ** 2, 0.0, None)
            sd_t = np.sqrt(var_t)
        mean_t = np.where(triggered[e], 0.0, mean_t)
        sd_t = np.where(triggered[e], 0.0, sd_t)
        out[("mean_t", e)] = mean_t
        out[("sd_t", e)] = sd_t
        out[("haz_k", e)] = step_haz[e]   # (n, horizon) raw per-year hazards
    return out
