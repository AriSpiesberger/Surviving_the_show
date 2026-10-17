"""Point-in-time 40-man roster features (MLB Stats API, sources/mlb_rosters).

For a player at as_of_year S:
    r40_on_nov   on a club's 40-man roster on Nov 30 of S (after the Rule 5
                 protection deadline). 1/0; NaN for S before the cache starts.
    r40_novs     number of Nov 30 snapshots <= S he was on a 40-man roster
                 (proxy for option years used without sticking in the majors).

Nov 30 of S is after season S ends and before season S+1 starts, so the flag
is known at every landmark's prediction time. Both features are populated for
every player in a covered season (absent from all rosters = 0), so coverage
does not depend on outcome.

OFF by default so the production feature contract is unchanged until an A/B
run says otherwise. Set ROSTER40_ON=1 to append them.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from prospects import config

_ON = os.environ.get("ROSTER40_ON") == "1"

ROSTER40_NAMES: list[str] = []
_SEASONS: dict[str, np.ndarray] = {}  # mlbam_id -> sorted seasons on a Nov 30 40-man
_FIRST = _LAST = None


def _load(path=None) -> None:
    global ROSTER40_NAMES, _SEASONS, _FIRST, _LAST
    path = path or config.ROSTER40_NOV
    d = pd.read_csv(path, usecols=["season", "mlbam_id"])
    _FIRST, _LAST = int(d.season.min()), int(d.season.max())
    _SEASONS = {str(int(k)): np.sort(g.season.unique())
                for k, g in d.drop_duplicates().groupby("mlbam_id")}
    ROSTER40_NAMES = ["r40_on_nov", "r40_novs"]
    import datetime as _dt
    if _LAST < _dt.date.today().year:
        print(f"[roster] WARNING: {path.name} stops at {_LAST}; every later landmark gets "
              f"NaN r40_* features. Run `python -m prospects.data.sources.mlb_rosters` "
              f"(it resumes) before an A/B run.")


if _ON and config.ROSTER40_NOV.exists():
    _load()


def _key(mlbam_id) -> str | None:
    if mlbam_id is None:
        return None
    try:
        if isinstance(mlbam_id, float) and np.isnan(mlbam_id):
            return None
        return str(int(mlbam_id))
    except (TypeError, ValueError):
        return None


def roster40_dict(mlbam_id, as_of_year) -> dict[str, float]:
    """{r40_*: value} for as_of_year; {} when the features are off."""
    if not ROSTER40_NAMES:
        return {}
    if as_of_year is None or as_of_year < _FIRST or as_of_year > _LAST:
        return {"r40_on_nov": np.nan, "r40_novs": np.nan}
    seasons = _SEASONS.get(_key(mlbam_id))
    if seasons is None:
        return {"r40_on_nov": 0.0, "r40_novs": 0.0}
    n = int(np.searchsorted(seasons, as_of_year, side="right"))
    on = 1.0 if n and seasons[n - 1] == as_of_year else 0.0
    return {"r40_on_nov": on, "r40_novs": float(n)}
