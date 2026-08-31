"""Technical feature selection over the landmark hazard panel.

The hazard feature vector (``features.scouting.FEATURE_NAMES``, 238 wide plus
the appended ``horizon_offset_k``) grew by accretion: every generation bolted
on another block — per-year lags, career-to-date, trajectory, deltas,
accelerations, window summaries, scouting grades — and nothing was ever taken
back out. Blocks overlap heavily by construction (``best_woba`` vs
``current_woba_vs_best_woba`` vs ``woba_yT``; five lags of the same stat), and
several columns are structurally dead for a given cohort (all-NaN, constant, or
populated for one side of the ball only).

This module answers one question, empirically and per event: **which of those
columns actually carry independent signal for the hazard heads?** It is
deliberately *technical* — no baseball judgment, only data-driven screens — and
it is deliberately **standalone**: it reads the panel cache a training run
already wrote, and emits a manifest. It does not modify the feature contract,
and nothing in the training path imports it yet. Wiring a manifest into
``fit_landmark_hazards`` is a separate, reviewable step; ``feature_mask`` below
exists for whoever takes it.

The screens, in order (each stage sees only what survived the previous one):

  1. **missingness**   — NaN fraction >= ``max_missing``. HistGBT handles NaN
     natively, so this is an availability screen, not a correctness one: a
     column present for 2% of landmarks cannot move a hazard.
  2. **degenerate**    — zero variance, or a single value covering
     >= ``max_dominant`` of the non-missing rows.
  3. **redundancy**    — greedy Spearman filter. Features are walked in
     descending univariate strength; any feature with |rho| >= ``max_rho``
     against an already-kept feature is dropped and recorded against the
     survivor that subsumed it. Greedy (not clustering) on purpose — single
     linkage chains through a correlated block and evicts things that are only
     transitively related.
  4. **univariate**    — |AUC - 0.5| < ``min_auc_lift``. A column with no
     marginal rank association *and* no interaction value will not survive
     stage 5 either; this is the cheap pre-filter.
  5. **permutation**   — permutation importance on the held-out cohort, using
     the real ``HistGradientBoostingClassifier`` at the real hazard HP, scored
     by average precision. The drop floor is not zero: shadow features
     (permuted copies of real columns, Boruta-style) are fitted alongside the
     candidates and the largest shadow importance defines the noise floor, so
     "positive but indistinguishable from a shuffled column" counts as a drop.

  6. **verification**  — refit full-set vs surviving-set on the same rows and
     report held-out AP/AUC/logloss. A selection that costs AP is a selection
     you do not ship; the manifest carries the delta so that call is explicit.

Leakage discipline: every statistic — missingness, correlation, univariate AUC,
the permutation model's fit — is computed on the **fit** cohort only. The
permutation and verification scores are the only things that touch **val**, and
they only ever read it. The split is the run's existing player-level
``fit_pids.txt`` / ``val_pids.txt``, so a player cannot straddle it.

Events default to the five curve events the joint layer actually consumes
(``joint.HAZARD_CURVE_EVENTS``): TOP_100_PROSPECT, MLB_DEBUT, ESTABLISHED_MLB,
ELITE, STAR. A feature is kept in the union manifest if it survives for **any**
event — the panel is shared, so a column that only matters for STAR still has
to be there.

Usage
-----
    python -m prospects.features.selection                    # runs/current
    python -m prospects.features.selection --events MLB_DEBUT ESTABLISHED_MLB
    python -m prospects.features.selection --skip-perm        # fast screens only
    python -m prospects.features.selection --tag cand --out somewhere.json

Outputs (default ``runs/<tag>/models/``):
    feature_selection.json        the manifest: keep/drop + reason + per-event
    feature_selection_stats.csv   one row per (event, feature) with every stat
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from prospects import config
from prospects.features.scouting import FEATURE_NAMES, N_FEATURES
from prospects.model import joint
from prospects.model.hazards import landmark as lm
from prospects.model.hazards.survival import ELITE_KEY, MAX_OBS_YEAR, STAR_KEY
from prospects.core.schema import CareerEvent


# --- screen thresholds -----------------------------------------------------
# Defaults are deliberately conservative: this module should propose drops that
# are obvious, not drops that are arguable. Tighten via the CLI to explore.
# Missingness drops columns that are NEVER present, and nothing else. An
# earlier 0.98 threshold cost 20% of held-out AP on MLB_DEBUT: the 98-99.4%
# NaN block (best_top100_rank, the scout_*_p grades, bonus_vs_slot) is the most
# valuable in the panel, because for a rare event "was ranked/scouted at all"
# is itself enormous signal and HistGBT reads it natively through the NaN.
# Rarity is not the same as uninformativeness — let stage 5 judge these.
MAX_MISSING = 1.0       # NaN fraction at or above which a column is dropped
MAX_DOMINANT = 0.995    # single-value share (of non-missing) that counts as degenerate
MAX_RHO = 0.98          # |Spearman| against a kept feature that counts as redundant
MIN_AUC_LIFT = 0.005    # |AUC - 0.5| below which a column has no marginal signal

# --- sampling caps ---------------------------------------------------------
# Event row counts run to millions ((landmark, k) triplets); every stage below
# is a screen, not a fit we ship, so all of them run on capped subsamples.
MAX_STAT_ROWS = 400_000     # rows for missingness / degeneracy / univariate AUC
MAX_CORR_ROWS = 60_000      # rows for the Spearman matrix (O(n * p) rank sort)
MAX_FIT_ROWS = 250_000      # rows for the permutation + verification model fits
MAX_EVAL_ROWS = 60_000      # held-out rows scored by permutation importance
N_SHADOW = 8                # permuted decoy columns defining the noise floor
PERM_REPEATS = 3

# The k column is structural — inference writes the horizon into it. It is never
# a selection candidate.
PROTECTED = (lm.LANDMARK_K_FEATURE,)

# String -> hazard event key, for --events.
_EVENT_BY_NAME: dict = {lm._ename(e): e for e in CareerEvent.all_events()}
_EVENT_BY_NAME["ELITE"] = ELITE_KEY
_EVENT_BY_NAME["STAR"] = STAR_KEY

DEFAULT_EVENTS = tuple(joint.HAZARD_CURVE_EVENTS)  # the curves the joint reads


# ==========================================================================
# Result containers
# ==========================================================================
@dataclass
class EventSelection:
    """One event's verdict over the shared feature vector."""

    event: str
    n_rows_train: int
    n_pos_train: int
    n_rows_val: int
    n_pos_val: int
    keep: list[str] = field(default_factory=list)
    drop: dict[str, str] = field(default_factory=dict)      # feature -> stage
    reason: dict[str, str] = field(default_factory=dict)    # feature -> detail
    stats: pd.DataFrame | None = None
    verification: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "n_rows_train": self.n_rows_train,
            "n_pos_train": self.n_pos_train,
            "n_rows_val": self.n_rows_val,
            "n_pos_val": self.n_pos_val,
            "n_keep": len(self.keep),
            "n_drop": len(self.drop),
            "keep": self.keep,
            "drop": [
                {"feature": f, "stage": self.drop[f],
                 "reason": self.reason.get(f, "")}
                for f in sorted(self.drop, key=lambda x: (self.drop[x], x))
            ],
            "verification": self.verification,
        }


# ==========================================================================
# Stage primitives (generic: any X / y / names)
# ==========================================================================
def _subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices, capped. Returns all of them (in order) when n <= cap."""
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))


def _missing_rate(X: np.ndarray) -> np.ndarray:
    return np.isnan(X).mean(axis=0)


def _dominant_share(X: np.ndarray) -> np.ndarray:
    """Share of the most common value among each column's non-missing rows.

    1.0 for a constant column; ~0 for a continuous one. NaN-only columns score
    1.0 so they read as degenerate rather than as a division by zero.
    """
    out = np.ones(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        _, counts = np.unique(col, return_counts=True)
        out[j] = counts.max() / col.size
    return out


def _univariate_auc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column ROC AUC against y, over that column's non-missing rows.

    Missingness is not itself scored here — a column that is informative only
    through its NaN pattern is a data-availability artifact, and stage 5 will
    pick it up if it genuinely matters.
    """
    out = np.full(X.shape[1], 0.5, dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = ~np.isnan(col)
        yy = y[ok]
        if ok.sum() < 100 or yy.min() == yy.max():
            continue
        try:
            out[j] = roc_auc_score(yy, col[ok])
        except ValueError:
            pass
    return out


def _spearman_matrix(X: np.ndarray) -> np.ndarray:
    """|Spearman| correlation matrix, median-imputing NaN before ranking.

    Imputation biases correlation toward the imputed mass, which for a
    redundancy screen is the safe direction: two columns missing on the same
    rows look *more* alike, so we are more likely to call them redundant, and
    stage 5 still has to clear whatever survives.
    """
    Z = X.astype(np.float64, copy=True)
    for j in range(Z.shape[1]):
        col = Z[:, j]
        miss = np.isnan(col)
        if miss.any():
            fill = np.nanmedian(col) if (~miss).any() else 0.0
            col[miss] = 0.0 if np.isnan(fill) else fill
    R = rankdata(Z, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(R, rowvar=False)
    return np.nan_to_num(np.abs(np.atleast_2d(C)), nan=0.0)


def _greedy_corr_prune(
    names: list[str], C: np.ndarray, order: list[int], max_rho: float,
) -> tuple[list[int], dict[int, tuple[str, float]]]:
    """Walk `order` (strongest first); drop anything too close to a survivor.

    Returns (kept positional indices, {dropped index: (survivor name, rho)}).
    """
    kept: list[int] = []
    dropped: dict[int, tuple[str, float]] = {}
    for i in order:
        redundant_with = None
        for j in kept:
            if C[i, j] >= max_rho:
                redundant_with = (names[j], float(C[i, j]))
                break
        if redundant_with is None:
            kept.append(i)
        else:
            dropped[i] = redundant_with
    return kept, dropped


def _fit_hazard(X: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    """The production hazard estimator at production HP — the screen has to
    judge features against the model that will actually consume them."""
    return HistGradientBoostingClassifier(
        **lm._HAZARD_HP_DEFAULTS, random_state=seed,
    ).fit(X, y)


def _shadow_block(
    X: np.ndarray, n_shadow: int, rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    """Boruta-style decoys: permuted copies of randomly chosen real columns.

    Permuting real columns (rather than drawing noise) keeps the marginal
    distribution and the NaN rate, so the decoys compete for splits on the same
    terms as their originals while carrying zero signal by construction.
    """
    n_shadow = min(n_shadow, X.shape[1])
    src = rng.choice(X.shape[1], size=n_shadow, replace=False)
    S = X[:, src].copy()
    for j in range(S.shape[1]):
        S[:, j] = S[rng.permutation(S.shape[0]), j]
    return S, [int(s) for s in src]


def _score_set(clf, X: np.ndarray, y: np.ndarray) -> dict:
    p = clf.predict_proba(X)[:, 1]
    out = {"n": int(y.size), "n_pos": int(y.sum())}
    if y.min() == y.max():
        return {**out, "ap": None, "auc": None, "logloss": None}
    return {
        **out,
        "ap": float(average_precision_score(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }


# ==========================================================================
# The selector
# ==========================================================================
def select_features(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    names: list[str],
    *,
    event: str = "event",
    protect: tuple[str, ...] = PROTECTED,
    max_missing: float = MAX_MISSING,
    max_dominant: float = MAX_DOMINANT,
    max_rho: float = MAX_RHO,
    min_auc_lift: float = MIN_AUC_LIFT,
    skip_perm: bool = False,
    skip_verify: bool = False,
    n_shadow: int = N_SHADOW,
    perm_repeats: int = PERM_REPEATS,
    seed: int = 42,
    verbose: bool = True,
) -> EventSelection:
    """Run the six-stage screen for one event. Pure function over arrays.

    ``X_tr``/``y_tr`` must be the fit cohort and ``X_va``/``y_va`` the held-out
    cohort — the caller owns that split, this function never re-splits.
    """
    rng = np.random.default_rng(seed)
    p = len(names)
    assert X_tr.shape[1] == p and X_va.shape[1] == p, "names / matrix width mismatch"

    sel = EventSelection(
        event=event,
        n_rows_train=int(y_tr.size), n_pos_train=int(y_tr.sum()),
        n_rows_val=int(y_va.size), n_pos_val=int(y_va.sum()),
    )
    protect_set = {n for n in protect if n in names}
    alive = np.ones(p, dtype=bool)

    def _kill(idx: int, stage: str, why: str) -> None:
        if names[idx] in protect_set:
            return
        alive[idx] = False
        sel.drop[names[idx]] = stage
        sel.reason[names[idx]] = why

    # --- stats subsample ---------------------------------------------------
    s_idx = _subsample(X_tr.shape[0], MAX_STAT_ROWS, rng)
    Xs, ys = X_tr[s_idx], y_tr[s_idx]

    # --- stage 1: missingness ---------------------------------------------
    miss = _missing_rate(Xs)
    for j in np.where(miss >= max_missing)[0]:
        _kill(int(j), "missingness", f"nan_frac={miss[j]:.4f}")
    if max_missing < 1.0 and verbose:
        print(f"    WARNING: max_missing={max_missing} drops rare-but-present "
              f"columns. On this panel the 0.98-1.0 NaN block is worth ~20% "
              f"of held-out AP; 1.0 (never-present only) is the safe rule.")

    # --- stage 2: degeneracy ----------------------------------------------
    dom = _dominant_share(Xs)
    for j in np.where(alive)[0]:
        if dom[j] >= max_dominant:
            _kill(int(j), "degenerate", f"dominant_share={dom[j]:.5f}")

    # --- univariate signal (feeds stages 3 and 4) --------------------------
    # Computed for EVERY column, including ones already killed: an AUC of 0.5
    # in the stats CSV then means "no marginal signal", not "not measured".
    # That distinction is what made the missingness misfire hard to see.
    auc = _univariate_auc(Xs, ys)
    lift = np.abs(auc - 0.5)

    # --- stage 3: redundancy ----------------------------------------------
    # Ordered by univariate strength so the survivor of a correlated block is
    # its strongest member, and protected columns lead so nothing can subsume
    # them.
    live = list(np.where(alive)[0])
    c_idx = _subsample(Xs.shape[0], MAX_CORR_ROWS, rng)
    C = _spearman_matrix(Xs[np.ix_(c_idx, live)])
    local_names = [names[j] for j in live]
    order = sorted(
        range(len(live)),
        key=lambda i: (local_names[i] not in protect_set,
                       -lift[live[i]], local_names[i]),
    )
    _, red = _greedy_corr_prune(local_names, C, order, max_rho)
    for i, (survivor, rho) in red.items():
        _kill(live[i], "redundant", f"rho={rho:.4f} with {survivor}")

    # --- stage 4: univariate floor ----------------------------------------
    for j in np.where(alive)[0]:
        if lift[j] < min_auc_lift:
            _kill(int(j), "univariate", f"auc={auc[j]:.4f}")

    # --- stage 5: permutation importance vs shadow floor -------------------
    perm_mean = np.full(p, np.nan)
    perm_std = np.full(p, np.nan)
    shadow_floor = None
    if not skip_perm and sel.n_pos_train >= 50 and sel.n_pos_val >= 20:
        live = list(np.where(alive)[0])
        f_idx = _subsample(X_tr.shape[0], MAX_FIT_ROWS, rng)
        e_idx = _subsample(X_va.shape[0], MAX_EVAL_ROWS, rng)
        Xf, yf = X_tr[np.ix_(f_idx, live)], y_tr[f_idx]
        Xe, ye = X_va[np.ix_(e_idx, live)], y_va[e_idx]
        Sf, src = _shadow_block(Xf, n_shadow, rng)
        Se = Xe[:, src].copy()
        for j in range(Se.shape[1]):
            Se[:, j] = Se[rng.permutation(Se.shape[0]), j]
        Xf_a = np.hstack([Xf, Sf])
        Xe_a = np.hstack([Xe, Se])
        if verbose:
            print(f"    permutation: fit {Xf_a.shape} -> score {Xe_a.shape} "
                  f"({len(live)} live + {Sf.shape[1]} shadow)", flush=True)
        clf = _fit_hazard(Xf_a, yf, seed)
        r = permutation_importance(
            clf, Xe_a, ye, scoring="average_precision",
            n_repeats=perm_repeats, random_state=seed, n_jobs=1,
        )
        for i, j in enumerate(live):
            perm_mean[j] = float(r.importances_mean[i])
            perm_std[j] = float(r.importances_std[i])
        shadow = r.importances_mean[len(live):]
        shadow_floor = float(shadow.max()) if shadow.size else 0.0
        for i, j in enumerate(live):
            if r.importances_mean[i] <= shadow_floor:
                _kill(int(j), "permutation",
                      f"imp={r.importances_mean[i]:.6g} <= shadow_floor="
                      f"{shadow_floor:.6g}")

    sel.keep = [names[j] for j in np.where(alive)[0]]

    # --- stage 6: verification --------------------------------------------
    if not skip_verify and sel.n_pos_train >= 50 and sel.n_pos_val >= 20:
        live = list(np.where(alive)[0])
        f_idx = _subsample(X_tr.shape[0], MAX_FIT_ROWS, rng)
        e_idx = _subsample(X_va.shape[0], MAX_EVAL_ROWS, rng)
        yf, ye = y_tr[f_idx], y_va[e_idx]
        full = _score_set(_fit_hazard(X_tr[f_idx], yf, seed), X_va[e_idx], ye)
        red_ = _score_set(
            _fit_hazard(X_tr[np.ix_(f_idx, live)], yf, seed),
            X_va[np.ix_(e_idx, live)], ye,
        )
        sel.verification = {
            "n_features_full": p, "n_features_selected": len(live),
            "full": full, "selected": red_,
            "delta_ap": (None if full["ap"] is None or red_["ap"] is None
                         else red_["ap"] - full["ap"]),
            "delta_auc": (None if full["auc"] is None or red_["auc"] is None
                          else red_["auc"] - full["auc"]),
        }
        if verbose:
            d = sel.verification["delta_ap"]
            print(f"    verify: AP {full['ap']:.5f} (full, {p}f) -> "
                  f"{red_['ap']:.5f} (selected, {len(live)}f)  "
                  f"delta={d:+.5f}", flush=True)

    sel.stats = pd.DataFrame({
        "event": event,
        "feature": names,
        "kept": alive,
        "drop_stage": [sel.drop.get(n, "") for n in names],
        "drop_reason": [sel.reason.get(n, "") for n in names],
        "nan_frac": miss,
        "dominant_share": dom,
        "auc": auc,
        "auc_lift": lift,
        "perm_importance": perm_mean,
        "perm_std": perm_std,
        "shadow_floor": shadow_floor,
    })
    return sel


# ==========================================================================
# Panel adapter — build (X, y) for one hazard event from the cached panel
# ==========================================================================
def panel_paths(tag: str | None = None, own: bool = False) -> tuple[Path, Path]:
    """(cache npz, meta pkl) for either the OOF pipeline's panel or our own.

    ``own=True`` points at ``scratch/selection/``, the cache ``--build-panel``
    writes. We never write into ``scratch/oof/``: that cache belongs to the OOF
    pipeline, which resumes from it, and a selection run has no business
    replacing an artifact a fold sequence may be mid-way through.
    """
    run = config.run(tag)
    d = run.scratch / ("selection" if own else "oof")
    return d / "panel_cache.npz", d / "panel_meta.pkl"


def load_panel(tag: str | None = None,
               panel_npz: str | Path | None = None) -> dict:
    """Load a landmark panel cache — the OOF run's by default.

    Returns the pieces ``fit_landmark_hazards`` consumes: X_lm at landmark
    granularity, per-landmark pids and landmark years, the joined prospect
    dicts, and the season-stats lookup the row expander censors against.

    Hard-fails on a width mismatch. The cache stores no feature names, so a
    matrix that is not exactly ``N_FEATURES`` wide cannot be aligned to
    ``FEATURE_NAMES`` — every column label past the first divergence would be
    wrong, and the screen would confidently drop the wrong features. A stale
    cache is common in practice: adding a feature block widens the contract
    without invalidating the file on disk.
    """
    if panel_npz is not None:
        npz_path = Path(panel_npz)
        meta_path = npz_path.with_name("panel_meta.pkl")
    else:
        npz_path, meta_path = panel_paths(tag)
        if not npz_path.exists():
            own_npz, own_meta = panel_paths(tag, own=True)
            if own_npz.exists():
                npz_path, meta_path = own_npz, own_meta
    if not npz_path.exists():
        raise FileNotFoundError(
            f"No panel cache at {npz_path}. Either run the OOF pipeline (it "
            f"writes one before the first fold) or build a selection-local "
            f"one with `python -m prospects.features.selection --build-panel`."
        )
    npz = np.load(npz_path, allow_pickle=True)
    X_lm = npz["X_lm"]
    if X_lm.shape[1] != N_FEATURES:
        raise ValueError(
            f"Panel cache {npz_path} is {X_lm.shape[1]} features wide but the "
            f"current contract (features.scouting.FEATURE_NAMES) is "
            f"{N_FEATURES}. The cache predates a feature-block change and "
            f"carries no column names, so its columns cannot be aligned. "
            f"Rebuild it: `python -m prospects.features.selection "
            f"--build-panel`, or rerun the OOF pipeline."
        )
    with meta_path.open("rb") as fh:
        meta = pickle.load(fh)
    joined_idx = npz["joined_idx"]
    return {
        "path": str(npz_path),
        "X_lm": X_lm,
        "pids": [str(p) for p in npz["pids"].tolist()],
        "S_yrs": [int(s) for s in npz["S_yrs"].tolist()],
        "joined": [meta["prospects"][i] for i in joined_idx],
        "stats_by_pid": meta["stats_by_pid"],
    }


def build_panel_cache(
    tag: str | None = None,
    *,
    max_draft_year: int = 2020,
    min_landmark_year: int = 2007,
    max_landmark_year: int | None = None,
    verbose: bool = True,
) -> Path:
    """Build a selection-local panel at the *current* feature contract.

    Same builder the training pipeline uses (``lm.build_landmark_panel``), same
    defaults, written to ``scratch/selection/`` so it can never be confused
    with — or clobber — the OOF pipeline's cache. Nothing here is partial-season
    augmented: the screen judges columns on the complete-season manifold.
    """
    from prospects.core.storage import ProspectDB

    npz_path, meta_path = panel_paths(tag, own=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    db = ProspectDB(str(config.MODEL_DB))
    t0 = time.time()
    X_lm, pids, S_yrs, joined, stats_by_pid = lm.build_landmark_panel(
        db, max_draft_year=max_draft_year,
        min_landmark_year=min_landmark_year,
        max_landmark_year=max_landmark_year,
        include_ifa=True, verbose=verbose,
    )
    # `joined` repeats a prospect dict per landmark; store it deduped plus an
    # index, matching the OOF cache's layout so load_panel reads either one.
    uniq: list[dict] = []
    seen: dict[int, int] = {}
    joined_idx = np.empty(len(joined), dtype=np.int32)
    for i, p in enumerate(joined):
        key = id(p)
        if key not in seen:
            seen[key] = len(uniq)
            uniq.append(p)
        joined_idx[i] = seen[key]
    np.savez_compressed(
        npz_path, X_lm=X_lm, pids=np.asarray(pids, dtype=object),
        S_yrs=np.asarray(S_yrs, dtype=np.int32), joined_idx=joined_idx,
    )
    with meta_path.open("wb") as fh:
        pickle.dump({"prospects": uniq, "stats_by_pid": stats_by_pid}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"panel built in {(time.time()-t0)/60:.1f} min  "
              f"X_lm={X_lm.shape}\n  wrote {npz_path}\n  wrote {meta_path}")
    return npz_path


def _read_pids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build the split with "
            f"`python -m prospects.model.train.make_split`."
        )
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def event_rows(panel: dict, event, max_obs_year: int = MAX_OBS_YEAR):
    """Expand the panel to this event's (landmark_idx, k, y) triplets, using
    the trainer's own eligibility + right-censoring policy so the screen sees
    exactly the rows the hazard fit would see."""
    ename = lm._ename(event)
    rc, min_yrs = lm.EVENT_POLICY_LM.get(ename, (True, 0))
    K = lm.K_PER_EVENT.get(event, 10)
    return lm.landmark_event_rows(
        panel["joined"], panel["S_yrs"], event, K, panel["stats_by_pid"],
        right_censor=rc, min_years_to_fire=min_yrs, max_obs_year=max_obs_year,
    )


def _gather(X_lm: np.ndarray, landmark_idx: np.ndarray,
            k_arr: np.ndarray) -> np.ndarray:
    """Materialize (n, N_FEATURES+1) for the given triplets — the same layout
    ``lm._assemble_event_X`` builds, but only for rows we sampled."""
    X = np.empty((landmark_idx.size, lm.N_FEATURES_LM), dtype=np.float32)
    X[:, :N_FEATURES] = X_lm[landmark_idx]
    X[:, lm.K_FEATURE_INDEX] = k_arr.astype(np.float32)
    return X


def build_event_matrices(
    panel: dict, event, fit_pids: set[str], val_pids: set[str],
    *, cap: int = MAX_STAT_ROWS, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X_tr, y_tr, X_va, y_va) for one event, split by the run's player split.

    Rows are capped on each side before materialization — the full expansion is
    millions of rows wide by 239 float32 and there is no reason to hold it.
    Landmarks whose player is in neither pid list are dropped, so the screen
    never reads a player the training split excluded.
    """
    rng = np.random.default_rng(seed)
    landmark_idx, k_arr, y = event_rows(panel, event)
    pids = np.asarray(panel["pids"])
    row_pid = pids[landmark_idx]
    in_fit = np.fromiter((p in fit_pids for p in row_pid), bool, row_pid.size)
    in_val = np.fromiter((p in val_pids for p in row_pid), bool, row_pid.size)

    out = []
    for mask, row_cap in ((in_fit, cap), (in_val, MAX_EVAL_ROWS * 4)):
        sub = np.where(mask)[0]
        if sub.size > row_cap:
            sub = np.sort(rng.choice(sub, size=row_cap, replace=False))
        out.append(_gather(panel["X_lm"], landmark_idx[sub], k_arr[sub]))
        out.append(y[sub].astype(np.int8))
    return out[0], out[1], out[2], out[3]


# ==========================================================================
# Driver
# ==========================================================================
def select_hazard_features(
    tag: str | None = None,
    events: tuple[str, ...] = DEFAULT_EVENTS,
    *,
    seed: int = 42,
    verbose: bool = True,
    panel_npz: str | Path | None = None,
    **screen_kwargs,
) -> tuple[dict, pd.DataFrame]:
    """Run the screen for each event and union the survivors.

    Returns (manifest dict, per-(event, feature) stats frame). The manifest's
    top-level ``keep`` is the union: the panel is shared across heads, so a
    column only leaves it if *every* event rejected it.
    """
    run = config.run(tag)
    panel = load_panel(tag, panel_npz)
    fit_pids = _read_pids(run.fit_pids)
    val_pids = _read_pids(run.val_pids)
    names = list(lm.FEATURE_NAMES_LM)
    if verbose:
        print(f"panel: X_lm={panel['X_lm'].shape}  "
              f"landmarks={len(panel['pids']):,}  features={len(names)}")
        print(f"split: fit={len(fit_pids):,} pids  val={len(val_pids):,} pids\n")

    per_event: dict[str, EventSelection] = {}
    frames: list[pd.DataFrame] = []
    for ename in events:
        if ename not in _EVENT_BY_NAME:
            raise KeyError(f"Unknown event {ename!r}. "
                           f"Known: {sorted(_EVENT_BY_NAME)}")
        t0 = time.time()
        if verbose:
            print(f"[{ename}] expanding rows...", flush=True)
        X_tr, y_tr, X_va, y_va = build_event_matrices(
            panel, _EVENT_BY_NAME[ename], fit_pids, val_pids, seed=seed)
        if verbose:
            print(f"    fit={X_tr.shape} pos={int(y_tr.sum()):,}  "
                  f"val={X_va.shape} pos={int(y_va.sum()):,}", flush=True)
        if y_tr.sum() < 10 or y_va.sum() < 5:
            print(f"    too few positives — skipping {ename}")
            continue
        sel = select_features(X_tr, y_tr, X_va, y_va, names, event=ename,
                              seed=seed, verbose=verbose, **screen_kwargs)
        per_event[ename] = sel
        frames.append(sel.stats)
        if verbose:
            print(f"    keep={len(sel.keep)}  drop={len(sel.drop)}  "
                  f"({time.time()-t0:.0f}s)\n", flush=True)
        del X_tr, y_tr, X_va, y_va

    if not per_event:
        raise RuntimeError("No event produced a selection.")

    union_keep = sorted(
        {f for s in per_event.values() for f in s.keep},
        key=names.index,
    )
    union_drop = [n for n in names if n not in set(union_keep)]
    # A column is only dropped from the shared panel if every event rejected
    # it; record the stage each one used so a drop is auditable.
    drop_detail = [
        {"feature": f,
         "stages": {ev: s.drop.get(f, "") for ev, s in per_event.items()},
         "reasons": {ev: s.reason.get(f, "") for ev, s in per_event.items()}}
        for f in union_drop
    ]
    manifest = {
        "schema": 1,
        "tag": tag or "current",
        "generated_by": "prospects.features.selection",
        "panel": panel["path"],
        "n_landmarks": len(panel["pids"]),
        "base_feature_names": names,
        "n_features_in": len(names),
        "n_features_kept": len(union_keep),
        "keep": union_keep,
        "drop": drop_detail,
        "events": {ev: s.to_dict() for ev, s in per_event.items()},
        "thresholds": {
            "max_missing": screen_kwargs.get("max_missing", MAX_MISSING),
            "max_dominant": screen_kwargs.get("max_dominant", MAX_DOMINANT),
            "max_rho": screen_kwargs.get("max_rho", MAX_RHO),
            "min_auc_lift": screen_kwargs.get("min_auc_lift", MIN_AUC_LIFT),
            "n_shadow": screen_kwargs.get("n_shadow", N_SHADOW),
            "perm_repeats": screen_kwargs.get("perm_repeats", PERM_REPEATS),
            "seed": seed,
        },
    }
    return manifest, pd.concat(frames, ignore_index=True)


# ==========================================================================
# Consumers
# ==========================================================================
def load_selection(path: str | Path) -> dict:
    """Read a manifest written by this module."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def feature_mask(manifest: dict, names: list[str] | None = None) -> np.ndarray:
    """Boolean mask over ``names`` (default: the manifest's own feature order).

    This is the hook for wiring a manifest into training —
    ``X[:, feature_mask(m)]`` — and the reason the manifest stores the full
    ``base_feature_names`` it was computed against: a mask applied to a
    different feature order would silently mis-select. Raises if the orders
    disagree.
    """
    base = list(manifest["base_feature_names"])
    if names is not None and list(names) != base:
        raise ValueError(
            "Feature order does not match the manifest's base_feature_names; "
            "the manifest is stale relative to the current feature contract.")
    keep = set(manifest["keep"])
    return np.array([n in keep for n in base], dtype=bool)


# ==========================================================================
# CLI
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Technical feature selection over the landmark hazard panel.")
    ap.add_argument("--tag", default=None, help="run tag (default: current)")
    ap.add_argument("--events", nargs="+", default=list(DEFAULT_EVENTS),
                    help=f"hazard events to screen (default: {' '.join(DEFAULT_EVENTS)})")
    ap.add_argument("--out", default=None,
                    help="manifest path (default: runs/<tag>/models/feature_selection.json)")
    ap.add_argument("--panel", default=None,
                    help="explicit panel_cache.npz (default: the run's OOF "
                         "cache, else the selection-local one)")
    ap.add_argument("--build-panel", action="store_true",
                    help="rebuild a selection-local panel at the current "
                         "feature contract before screening")
    ap.add_argument("--max-draft-year", type=int, default=2020,
                    help="--build-panel: cohort cutoff")
    ap.add_argument("--min-landmark-year", type=int, default=2007,
                    help="--build-panel: earliest landmark season")
    ap.add_argument("--max-landmark-year", type=int, default=None,
                    help="--build-panel: latest landmark season")
    ap.add_argument("--max-missing", type=float, default=MAX_MISSING)
    ap.add_argument("--max-dominant", type=float, default=MAX_DOMINANT)
    ap.add_argument("--max-rho", type=float, default=MAX_RHO)
    ap.add_argument("--min-auc-lift", type=float, default=MIN_AUC_LIFT)
    ap.add_argument("--n-shadow", type=int, default=N_SHADOW)
    ap.add_argument("--perm-repeats", type=int, default=PERM_REPEATS)
    ap.add_argument("--skip-perm", action="store_true",
                    help="cheap screens only — no model fits for importance")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the full-vs-selected refit comparison")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run = config.run(args.tag)
    out = Path(args.out) if args.out else run.models / "feature_selection.json"
    stats_out = out.with_name(out.stem + "_stats.csv")

    t0 = time.time()
    panel_npz = args.panel
    if args.build_panel:
        panel_npz = build_panel_cache(
            args.tag, max_draft_year=args.max_draft_year,
            min_landmark_year=args.min_landmark_year,
            max_landmark_year=args.max_landmark_year,
        )

    manifest, stats = select_hazard_features(
        args.tag, tuple(args.events), seed=args.seed, panel_npz=panel_npz,
        max_missing=args.max_missing, max_dominant=args.max_dominant,
        max_rho=args.max_rho, min_auc_lift=args.min_auc_lift,
        n_shadow=args.n_shadow, perm_repeats=args.perm_repeats,
        skip_perm=args.skip_perm, skip_verify=args.skip_verify,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats.to_csv(stats_out, index=False)

    print("=" * 72)
    print(f"kept {manifest['n_features_kept']}/{manifest['n_features_in']} "
          f"features (union over {len(manifest['events'])} events)")
    for ev, e in manifest["events"].items():
        v = e.get("verification") or {}
        d = v.get("delta_ap")
        tail = f"  AP delta {d:+.5f}" if d is not None else ""
        print(f"  {ev:<20} keep={e['n_keep']:>4}  drop={e['n_drop']:>4}{tail}")
    by_stage: dict[str, int] = {}
    for d in manifest["drop"]:
        for st in d["stages"].values():
            if st:
                by_stage[st] = by_stage.get(st, 0) + 1
    if by_stage:
        print("  union drops by stage (counted per event): "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_stage.items())))
    print(f"\nwrote {out}\n      {stats_out}")
    print(f"total {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
