"""Down-sample a COMPLETE season into a PARTIAL (in-progress) one for training.

This is the training-time inverse of ``prorate.py``. At INFERENCE, prorate
inflates a live partial season UP to full-season equivalents (interpolation)
so a mid-season snapshot reads in-distribution to a model trained only on
complete seasons. The cleaner fix is to instead TRAIN on partial seasons: take
complete historical seasons and deflate them DOWN to in-progress lines, so the
hazard panel actually sees the mid-season feature manifold.

Two rules (from spec):

  1. Promotion within a season (e.g. A -> AA): the player FINISHED the lower
     stint before being promoted, so the lower-level line is COMPLETE and only
     the CURRENT (highest level reached) line is in-progress. We therefore
     down-sample ONLY the highest-rank level row of the snapshot season; every
     lower-level row that season — and every other season — passes through
     untouched.

  2. SAMPLE, don't interpolate. Partial counting stats are DRAWN from their
     sampling distribution at the reduced sample size (Binomial for per-PA
     events, Poisson for per-IP events), not linearly scaled. A linearly
     scaled half-season is impossibly smooth; real half-seasons are streaky.
     Sampling reproduces that in-season player variance so the model learns
     the true partial-season noise instead of a clean fraction of the whole.

Data note: ``season_stats`` stores RATE stats (avg/obp/slg/woba/iso/babip/
k_pct/bb_pct and era/fip/k9/bb9/hr9/whip) plus a few counts (pa, home_runs,
stolen_bases, ip). We reconstruct the coherent core events we can (BB, K, H,
HR, SB for hitters; K, BB, HR, ER, H for pitchers) by drawing them at the
partial sample size from the full-season rates, then recompute every rate
feature from those draws. No full box score (AB/2B/3B) is stored, so a couple
of derived rates (iso, babip) are drawn with a matched-variance proxy rather
than from an exact event count; this is documented inline.

Usage:
    from numpy.random import default_rng
    rng = default_rng(seed)
    partial = sample_partial_season(stats_rows, season_year=2023, frac=0.45, rng=rng)
    # or drive the fraction off a calendar date, per-level (mirrors prorate):
    partial = sample_partial_season(stats_rows, 2023, as_of=date(2023,6,11), rng=rng)
"""
from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Optional

import numpy as np

from prospects.features.prorate import _elapsed_fraction
from prospects.features.scouting import LEVEL_RANK

# Training/eval augmentation policy (see maybe_partial): per (player, year),
# with probability PARTIAL_RATE the snapshot season is down-sampled to a
# partial in-progress line whose elapsed fraction is drawn uniformly from
# FRAC_RANGE. Applied identically to train and val so the model learns — and
# is evaluated on — the real mid-season manifold, not only complete seasons.
PARTIAL_RATE = 0.5
FRAC_RANGE = (0.15, 0.85)

# Floors: below these a partial line is too thin to carry a rate signal, so we
# DROP the in-progress current-level row (the player simply "hasn't played
# enough yet at this level" — which is itself a realistic mid-season state).
_MIN_PA = 5
_MIN_IP = 2.0
_C_FIP = 3.10  # matches prospects.features.scouting._fip_value


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _stable_rng(player_id: str, year: int, seed: int) -> np.random.Generator:
    """Deterministic per-(player, year) RNG so a landmark gets the SAME
    partial/complete decision and fraction everywhere it appears (panel build
    and scoring), and runs reproduce. Python's hash() is salted per process,
    so derive the seed from a stable digest instead."""
    h = hashlib.blake2b(f"{player_id}|{year}|{seed}".encode(),
                        digest_size=8).hexdigest()
    return np.random.default_rng(int(h, 16))


def maybe_partial(
    stats_rows: list[dict],
    year: int,
    player_id: str,
    seed: int,
    *,
    rate: float = PARTIAL_RATE,
    frac_range: tuple[float, float] = FRAC_RANGE,
) -> list[dict]:
    """With probability `rate`, return stats with `year`'s current-level stint
    down-sampled to a uniform-random partial season; otherwise return the rows
    unchanged (complete). Deterministic in (player_id, year, seed)."""
    rng = _stable_rng(player_id, year, seed)
    if rng.random() >= rate:
        return stats_rows
    frac = float(rng.uniform(*frac_range))
    return sample_partial_season(stats_rows, year, frac=frac, rng=rng)


def partial_for_features(
    stats_rows: list[dict],
    season_year: int,
    player_id: str,
    partial_seed: Optional[int],
) -> list[dict]:
    """Single training-time integration hook for partial-season augmentation.

    Call this immediately before building a feature vector whose as-of /
    current season is ``season_year``. When ``partial_seed`` is None this is a
    strict no-op (returns ``stats_rows`` unchanged), so wiring it into a panel
    build or scorer is byte-for-byte identical to the old path until a seed is
    supplied. With a seed it applies the (player, year, seed)-deterministic
    50/50 down-sample of the current-season stint (see maybe_partial), so the
    SAME landmark gets the SAME partial/complete decision everywhere it appears
    (panel build, fit/val scoring, OOF scoring) as long as the same seed is
    threaded through.
    """
    if partial_seed is None:
        return stats_rows
    return maybe_partial(stats_rows, season_year, player_id, partial_seed)


def _sample_hitter(r: dict, frac: float, rng: np.random.Generator) -> Optional[dict]:
    """Down-sample one complete hitter season-row to `frac`. Returns a new row
    or None if the partial line falls below the sample floor."""
    P = float(r.get("pa") or 0)
    n = int(round(frac * P))
    if n < _MIN_PA:
        return None

    avg = _clip01(float(r["avg"])) if r.get("avg") is not None else None
    obp = _clip01(float(r["obp"])) if r.get("obp") is not None else None
    slg = float(r["slg"]) if r.get("slg") is not None else None
    iso = float(r["iso"]) if r.get("iso") is not None else (
        (slg - avg) if (slg is not None and avg is not None) else None)
    kp = _clip01(float(r["k_pct"])) if r.get("k_pct") is not None else None
    bbp = _clip01(float(r["bb_pct"])) if r.get("bb_pct") is not None else None
    babip = _clip01(float(r["babip"])) if r.get("babip") is not None else None
    hr_full = float(r.get("home_runs") or 0)
    sb_full = float(r.get("stolen_bases") or 0)

    out = dict(r)
    out["pa"] = float(n)

    # Draw the coherent core events at the partial sample size.
    bb = rng.binomial(n, bbp) if bbp is not None else 0
    k = rng.binomial(n, kp) if kp is not None else 0
    ab = max(n - bb, 1)  # ignore HBP/SF (not stored)
    h = rng.binomial(ab, avg) if avg is not None else None
    hr = rng.binomial(n, _clip01(hr_full / P)) if P > 0 else 0
    sb = rng.binomial(n, _clip01(sb_full / P)) if P > 0 else 0

    out["home_runs"] = float(hr)
    out["stolen_bases"] = float(sb)
    out["hr_per_pa"] = hr / n
    out["sb_per_pa"] = sb / n
    if bbp is not None:
        out["bb_pct"] = bb / n
    if kp is not None:
        out["k_pct"] = k / n
    if h is not None:
        out["avg"] = h / ab
        out["obp"] = (h + bb) / n  # HBP/SF ignored, consistent with ab above
        # ISO has no stored XBH breakdown; draw extra-base events with matched
        # variance (Binomial(AB, iso_full)) so power scales realistically with
        # the smaller sample instead of staying pinned at the full-season value.
        if iso is not None:
            xb = rng.binomial(ab, _clip01(iso))
            out["iso"] = xb / ab
            out["slg"] = out["avg"] + out["iso"]
        # BABIP: balls in play = AB - K - HR; redraw hits-on-contact.
        if babip is not None:
            bip = max(ab - k - hr, 1)
            out["babip"] = rng.binomial(bip, babip) / bip
        # wOBA proxy (same formula as scouting._woba_value).
        if out.get("obp") is not None and out.get("iso") is not None:
            out["woba"] = 0.55 * out["obp"] + 0.45 * (out["avg"] + 1.65 * out["iso"])
    out["season_complete"] = 0
    gp = r.get("games_played")
    if gp is not None:
        out["games_played"] = int(round(frac * gp))
    return out


def _sample_pitcher(r: dict, frac: float, rng: np.random.Generator) -> Optional[dict]:
    """Down-sample one complete pitcher season-row to `frac`. Returns a new row
    or None if below the IP floor."""
    I = float(r.get("ip") or 0)
    ip = frac * I
    if ip < _MIN_IP:
        return None

    def rate(key):
        return float(r[key]) if r.get(key) is not None else None
    k9, bb9, hr9, era, whip = (rate("k9"), rate("bb9"), rate("hr9"),
                               rate("era"), rate("whip"))

    out = dict(r)
    out["ip"] = ip
    # Per-IP events ~ Poisson at the partial innings.
    K = rng.poisson(max(k9, 0) * ip / 9.0) if k9 is not None else None
    BB = rng.poisson(max(bb9, 0) * ip / 9.0) if bb9 is not None else None
    HR = rng.poisson(max(hr9, 0) * ip / 9.0) if hr9 is not None else None
    ER = rng.poisson(max(era, 0) * ip / 9.0) if era is not None else None
    # Hits: H/IP = whip - bb9/9. Redraw if both available.
    H = None
    if whip is not None and bb9 is not None:
        h_rate = max(whip - bb9 / 9.0, 0.0)
        H = rng.poisson(h_rate * ip)

    if K is not None:
        out["k9"] = K * 9.0 / ip
    if BB is not None:
        out["bb9"] = BB * 9.0 / ip
    if HR is not None:
        out["hr9"] = HR * 9.0 / ip
    if ER is not None:
        out["era"] = ER * 9.0 / ip
    if H is not None and BB is not None:
        out["whip"] = (H + BB) / ip
    # FIP from the sampled components (matches scouting._fip_value identity).
    if K is not None and BB is not None and HR is not None:
        out["fip"] = (13.0 * HR + 3.0 * BB - 2.0 * K) / ip + _C_FIP
    # velo_avg is a measurement, not an accumulation -> unchanged.
    out["season_complete"] = 0
    gp = r.get("games_played")
    if gp is not None:
        out["games_played"] = int(round(frac * gp))
    return out


def sample_partial_season(
    stats_rows: list[dict],
    season_year: int,
    *,
    frac: Optional[float] = None,
    as_of: Optional[_dt.date] = None,
    rng: Optional[np.random.Generator] = None,
    drop_thin: bool = True,
) -> list[dict]:
    """Return a copy of stats_rows with the snapshot season's CURRENT-level
    line down-sampled to an in-progress partial season.

    Exactly one of `frac` or `as_of` must be given:
      - frac: explicit fraction of a full season already played (0..1).
      - as_of: a calendar date; the current level's elapsed fraction is taken
        from prorate's per-level season windows (mirrors inference proration).

    Rule 1 (promotion): only the HIGHEST level reached in `season_year` is
    sampled; lower-level stints that season are kept COMPLETE, as are all
    other seasons.

    Rule 2 (sampling): the current-level counts are drawn at the reduced
    sample size, not scaled (see _sample_hitter / _sample_pitcher).

    If the current-level partial line is below the sample floor it is dropped
    when drop_thin=True (a realistic "barely started at this level" state).
    """
    if (frac is None) == (as_of is None):
        raise ValueError("pass exactly one of frac= or as_of=")
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (rng=) for reproducibility")

    season_rows = [s for s in stats_rows if s.get("season_year") == season_year]
    if not season_rows:
        return [dict(s) for s in stats_rows]

    # Current stint = the highest-rank level reached this season.
    cur_rank = max(LEVEL_RANK.get((s.get("level") or "").upper(), 0)
                   for s in season_rows)

    out: list[dict] = []
    sampled_current = False
    for s in stats_rows:
        if s.get("season_year") != season_year:
            out.append(dict(s))
            continue
        rank = LEVEL_RANK.get((s.get("level") or "").upper(), 0)
        if rank != cur_rank or sampled_current:
            # Lower-level (completed) stint, or a duplicate of the current
            # rank — pass through COMPLETE.
            out.append(dict(s))
            continue
        sampled_current = True  # only sample the first current-level row
        f = frac if frac is not None else _elapsed_fraction(
            s.get("level") or "", as_of, season_year)
        if f >= 1.0:
            out.append(dict(s))     # season already complete at this date
            continue
        f = max(f, 0.0)
        is_pitcher = float(s.get("ip") or 0) > 0 and float(s.get("pa") or 0) == 0
        sampler = _sample_pitcher if is_pitcher else _sample_hitter
        # Two-way / ambiguous rows: sample whichever side carries the sample.
        partial = sampler(s, f, rng)
        if partial is None and float(s.get("ip") or 0) > 0:
            partial = _sample_pitcher(s, f, rng)
        if partial is None:
            if not drop_thin:
                out.append(dict(s))
            # else: drop the too-thin current stint entirely
            continue
        out.append(partial)
    return out
