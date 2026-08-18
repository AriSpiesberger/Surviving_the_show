"""
prospects/features/percentiles.py
=================================

Within-cohort percentile ranks, where a cohort is one (level, season_year).

Why this exists: essentially every rate stat in the panel is non-stationary
year to year, so a gradient-boosted tree splitting on a raw threshold is
selecting a different population depending on the season. Measured over
2005-2025, peak-to-trough movement of the yearly mean, in units of the
within-year sd:

    pitches_per_bf  8.0      k9      1.6      iso   1.0
    swing_pct       4.6      k_pct   1.5      era   0.8
    swstr_pct       3.6      hr9     1.2      woba  0.7

Two different causes, one fix. Some of it is the game changing (strikeout
rates climbing for two decades). Some is measurement coverage: the pitch-level
fields simply were not recorded at every level in the early years, so their
league mean moves when reporting improves rather than when players do. A
tree cannot correct for either — unlike a network, it cannot learn "subtract
this year's league mean" as an internal transformation. Ranking a player
against the peers they actually played against removes both.

No look-ahead: a cohort is a single (level, season_year), so a row is only
ever ranked against rows from its own season and its own level. Nothing from a
future season enters, which keeps this safe for walk-forward backtests. The
in-progress season ranks against its own peers-to-date, which is the correct
comparison for live scoring.

Percentiles are raw — higher value gives a higher percentile, with no
direction flipping for stats where lower is better (K% for a hitter, ERA for
a pitcher). Trees are indifferent to monotone direction, and flipping some
metrics and not others is an easy thing to get subtly wrong.

These land as `pct_*` columns on season_stats, written by
data/backfills/percentile_backfill.py. Storing them costs a staleness risk —
a percentile is only correct for the cohort it was computed over, and the
in-progress season gains rows on every pull — which is why the backfill runs
as a refresh step immediately after `pull`.

The alternative, attaching them in memory at panel-build time, avoids that
risk but loses a worse bet: nine separate call sites do
`SELECT * FROM season_stats` and group by player_id, including the live
scoring path in model/pipelines/prod.py. Attaching in memory means attaching
in all nine, and missing one is silent — the model trains with the columns
and scores without them, or vice versa. Columns cannot be missed.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from prospects.features.advanced import hitter_advanced, pitcher_advanced

# A cohort smaller than this does not support a meaningful rank — the early
# years at some levels have only a handful of qualified rows.
MIN_COHORT = 30

# Rows below these thresholds do not define the reference distribution: a
# 12-PA line is noise, and letting it set the shape of the cohort would move
# every other player's rank. Such rows are still ranked, against the reference
# the qualified rows define.
QUAL_PA = 50
QUAL_IP = 10


def _hr_per_pa(r: dict) -> Optional[float]:
    pa = r.get("pa") or 0
    hr = r.get("home_runs")
    return (hr / pa) if (pa > 0 and hr is not None) else None


def _sb_per_pa(r: dict) -> Optional[float]:
    pa = r.get("pa") or 0
    sb = r.get("stolen_bases")
    return (sb / pa) if (pa > 0 and sb is not None) else None


def _col(name: str) -> Callable[[dict], Optional[float]]:
    def get(r: dict) -> Optional[float]:
        v = r.get(name)
        return None if v is None else float(v)
    return get


# Derived metrics live in features/advanced.py and are not stored as columns.
# Computing one costs the whole dict, so the dict is computed once per row and
# stashed under a private key; the extractors below just read it. Without the
# memo, 16 metrics means 16 full recomputations of the same row.
_H_MEMO = "_adv_h"
_P_MEMO = "_adv_p"


def _adv(memo: str, key: str) -> Callable[[dict], Optional[float]]:
    def get(r: dict) -> Optional[float]:
        d = r.get(memo)
        if d is None:
            return None
        v = d.get(key)
        return None if v is None else float(v)
    return get


def _memoize_advanced(rows: list[dict]) -> None:
    for r in rows:
        if (r.get("pa") or 0) > 0:
            r[_H_MEMO] = hitter_advanced(r)
        if (r.get("ip") or 0) > 0:
            r[_P_MEMO] = pitcher_advanced(r)


def _drop_memos(rows: list[dict]) -> None:
    for r in rows:
        r.pop(_H_MEMO, None)
        r.pop(_P_MEMO, None)


# (feature name, extractor). The name becomes `pct_<name>` on the row.
#
# The stored-column metrics carry both a raw feature and this percentile in the
# panel. The derived block below is percentile-ONLY, deliberately: those are
# the most non-stationary quantities measured (pitches_per_bf moves 8.0
# within-year sd peak to trough, swing_pct 4.6, swstr_pct 3.6), so the raw
# number is the part that does not transfer across seasons. Rank against the
# peers actually faced is the signal; the absolute value is mostly era.
HIT_METRICS: list[tuple[str, Callable[[dict], Optional[float]]]] = [
    ("woba", _col("woba")),
    ("iso", _col("iso")),
    ("k_pct", _col("k_pct")),
    ("bb_pct", _col("bb_pct")),
    ("avg", _col("avg")),
    ("obp", _col("obp")),
    ("slg", _col("slg")),
    ("hr_per_pa", _hr_per_pa),
    ("sb_per_pa", _sb_per_pa),
    # --- derived, percentile-only ---
    ("swstr_pct", _adv(_H_MEMO, "swstr_pct")),
    ("contact_pct", _adv(_H_MEMO, "contact_pct")),
    ("pitches_per_pa", _adv(_H_MEMO, "pitches_per_pa")),
    ("gb_pct", _adv(_H_MEMO, "gb_pct")),
    ("fb_pct", _adv(_H_MEMO, "fb_pct")),
    ("ld_pct", _adv(_H_MEMO, "ld_pct")),
    ("hr_per_fb", _adv(_H_MEMO, "hr_per_fb")),
    ("babip", _adv(_H_MEMO, "babip")),
]

PIT_METRICS: list[tuple[str, Callable[[dict], Optional[float]]]] = [
    ("era", _col("era")),
    ("k9", _col("k9")),
    ("bb9", _col("bb9")),
    ("fip", _col("fip")),
    ("whip", _col("whip")),
    ("hr9", _col("hr9")),
    # --- derived, percentile-only ---
    ("p_swstr_pct", _adv(_P_MEMO, "swstr_pct")),
    ("p_contact_pct", _adv(_P_MEMO, "contact_pct")),
    ("p_strike_pct", _adv(_P_MEMO, "strike_pct")),
    ("p_pitches_per_bf", _adv(_P_MEMO, "pitches_per_bf")),
    ("p_gb_pct", _adv(_P_MEMO, "gb_pct")),
    ("p_fb_pct", _adv(_P_MEMO, "fb_pct")),
    ("p_ld_pct", _adv(_P_MEMO, "ld_pct")),
    ("p_k_bb_ratio", _adv(_P_MEMO, "k_bb_ratio")),
    ("p_babip_against", _adv(_P_MEMO, "babip_against")),
    ("p_siera_proxy", _adv(_P_MEMO, "siera_proxy")),
]

HIT_PCT_NAMES = [f"pct_{n}" for n, _ in HIT_METRICS]
PIT_PCT_NAMES = [f"pct_{n}" for n, _ in PIT_METRICS]


def _percentiles(ref: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Midrank percentile of each value against the reference distribution.

    Midrank (averaging the left and right insertion points) keeps ties at the
    centre of the block they occupy. Without it, a stat with a big spike at a
    single value — 0 home runs, 0 stolen bases — would put every tied player at
    the bottom of the block and read as far worse than the tie deserves.
    """
    lo = np.searchsorted(ref, vals, side="left")
    hi = np.searchsorted(ref, vals, side="right")
    return (lo + hi) / (2.0 * len(ref))


def attach_percentiles(rows: list[dict], verbose: bool = False) -> dict:
    """Add `pct_<metric>` keys to each row, in place.

    `rows` is every season_stats row — the whole table, not one player's — so
    that each cohort is complete. Rows whose cohort is too small, or whose own
    value is missing, simply do not get the key; the caller treats an absent
    key the same as any other missing feature.
    """
    _memoize_advanced(rows)
    cohorts: dict[tuple, list[dict]] = {}
    for r in rows:
        cohorts.setdefault((r.get("level"), r.get("season_year")), []).append(r)

    stats = {"cohorts": 0, "skipped_small": 0, "values": 0}
    for key, group in cohorts.items():
        for metrics, qual_key, qual_min in (
            (HIT_METRICS, "pa", QUAL_PA),
            (PIT_METRICS, "ip", QUAL_IP),
        ):
            qualified = [r for r in group if (r.get(qual_key) or 0) >= qual_min]
            if len(qualified) < MIN_COHORT:
                stats["skipped_small"] += 1
                continue
            stats["cohorts"] += 1
            for name, extract in metrics:
                ref = np.array(sorted(v for v in (extract(r) for r in qualified)
                                      if v is not None and np.isfinite(v)))
                if len(ref) < MIN_COHORT:
                    continue
                # Rank the whole group, not just the qualified subset, against
                # the reference the qualified rows define.
                targets, vals = [], []
                for r in group:
                    v = extract(r)
                    if v is not None and np.isfinite(v):
                        targets.append(r)
                        vals.append(v)
                if not targets:
                    continue
                pct = _percentiles(ref, np.array(vals))
                col = f"pct_{name}"
                for r, p in zip(targets, pct):
                    r[col] = float(p)
                stats["values"] += len(targets)

    _drop_memos(rows)
    if verbose:
        print(f"[percentiles] {stats['cohorts']:,} cohorts ranked, "
              f"{stats['skipped_small']:,} too small (<{MIN_COHORT}), "
              f"{stats['values']:,} values assigned")
    return stats
