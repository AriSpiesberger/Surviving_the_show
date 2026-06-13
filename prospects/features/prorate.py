"""Pro-rate an in-progress season's stat rows to full-season equivalents.

Mid-season scoring problem: the hazard panel trains exclusively on COMPLETE
seasons, so a June snapshot's 180-PA line reads to the model as an injury /
bench season — the exact profile it learned to bury. Interpolating the
counting stats to a full-season equivalent keeps the feature vector
in-distribution. Rate stats are left alone (they are what they are,
small-sample noise and all).

Guardrails:
  - Per-level calendars: on June 11 the full-season leagues are ~40%%
    elapsed but the DSL started last week. Elapsed fraction is computed
    against each level group's season window, not a global factor.
  - Minimum sample: rows with PA < 50 and IP < 15 are DROPPED, not scaled —
    multiplying 20 DSL PA by 6x manufactures a noise season whose rates
    then masquerade as a full year of evidence.
  - Scale factor capped at 4x regardless of calendar.

Apply ONLY to the live in-progress season at scoring time. Historical
snaps' season_year == snap_year rows are complete seasons — pro-rating
them would corrupt walk-forward features, so this is opt-in per call.
"""
from __future__ import annotations

import datetime as _dt

# Counting accumulators in season_stats. Everything else in the row is a
# rate (avg/obp/era/k_pct/...) or metadata and must not be scaled.
COUNTING_COLS = ("pa", "ip", "home_runs", "stolen_bases")

# Approximate (start, end) of each level group's season as (month, day).
_FULL_SEASON = ((4, 3), (9, 20))    # AAA / AA / A+ / A / A-
_COMPLEX = ((5, 4), (8, 28))        # ACL / FCL / CPX / RK complexes
_DSL = ((6, 2), (8, 26))            # Dominican Summer League

_MIN_PA = 50.0
_MIN_IP = 15.0
_MAX_FACTOR = 4.0


def _season_window(level: str) -> tuple[tuple[int, int], tuple[int, int]]:
    lv = (level or "").upper()
    if lv == "DSL":
        return _DSL
    if lv in ("RK", "CPX", "FCL", "ACL", "ROK"):
        return _COMPLEX
    return _FULL_SEASON


def _elapsed_fraction(level: str, as_of: _dt.date, year: int) -> float:
    (m0, d0), (m1, d1) = _season_window(level)
    start = _dt.date(year, m0, d0)
    end = _dt.date(year, m1, d1)
    total = (end - start).days
    if total <= 0:
        return 1.0
    return (as_of - start).days / total


def prorate_partial_season(
    stats_rows: list[dict],
    season_year: int,
    as_of: _dt.date,
) -> list[dict]:
    """Return a copy of stats_rows with the in-progress season's counting
    stats scaled to full-season equivalents.

    Rows from other seasons pass through untouched. In-progress rows below
    the minimum-sample floor are dropped. Rows are shallow-copied before
    mutation so the caller's stats_by_pid stays pristine.
    """
    out: list[dict] = []
    for s in stats_rows:
        if s.get("season_year") != season_year:
            out.append(s)
            continue
        frac = _elapsed_fraction(s.get("level") or "", as_of, season_year)
        if frac >= 1.0:
            out.append(s)          # season already complete
            continue
        pa = float(s.get("pa") or 0)
        ip = float(s.get("ip") or 0)
        if frac <= 0.0 or (pa < _MIN_PA and ip < _MIN_IP):
            continue               # not enough season to extrapolate
        factor = min(1.0 / frac, _MAX_FACTOR)
        r = dict(s)
        for col in COUNTING_COLS:
            v = r.get(col)
            if v is not None:
                r[col] = float(v) * factor
        out.append(r)
    return out
