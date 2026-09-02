"""Organizational depth charts: franchise resolution and "who is above you".

Two separate things live here, and they have very different value.

**1. Franchise resolution** (``resolve_franchise``) — reusable, and it closes a
real gap. ``season_stats.org`` is not one thing: at MiLB levels it is the
*affiliate* code (TOL, OKC, DUR) and at MLB it is the *parent club* (NYY, HOU).
That mismatch is why ``features.scouting`` dropped ``n_org_changes`` with the
note "season_stats.org is the MiLB affiliate not MLB org". This module closes it
with ``reference/baseballcube/affiliate_org_map.csv``, which is keyed by
(season, abbrev) and so stays correct across affiliation changes — including the
2021 MiLB reorganisation. Resolution is ~100% on affiliated MiLB, 89% at MLB
(the remainder are NULL-org rows).

Two traps handled here, both of which corrupt the join silently:

* MLB codes must NOT go through the affiliate map. Several collide with Negro
  League and historical clubs — ``COL`` matches "Columbus Buckeyes", ``PIT``
  matches "Pittsfield Cubs". MLB uses an explicit code table instead.
* ~10k AAA rows are Mexican League clubs (MTY, YUC, TIJ, PUE, ...) — AAA-
  classified but unaffiliated. They resolve to None and should be dropped, not
  treated as missing data.

**2. Blocking / "upward pressure" features** (``depth_features``) — built and
validated, and **they do not predict**. Recorded so the experiment is not
silently repeated.

The hypothesis was that reaching and sticking in MLB is driven more by
organisational opportunity (who is ahead of you on the depth chart) than by
individual performance. Tested over 2005-2022 MiLB player-seasons against
debut-within-3y, establish-within-6y, and establish-conditional-on-debuting::

    feature family                          AUC (AAA / AA)
    own performance percentile              0.66 / 0.68
    age                                     0.34 / 0.29   (inverted: strong)
    ALL blocking measures, pos-normalised   0.47 - 0.54

Holding position fixed, a 2-D read at AAA moves the debut rate +38 points across
performance quartiles and -3 points across blocking terciles. The blocking axis
is flat.

The likely reason is visible in ``org_change_rate``: **AAA players change
organisation ~50% of the time year over year** (AA 27%, A+ 17%, RK 8%) across
all rows, and still 39% when restricted to pre-debut players with a real
workload (>=50 PA / 20 IP) — the subset the tests above ran on. Either way the
population most exposed to blocking is the least attached to any one org, so a
block tends to resolve by movement rather than by suppressing the outcome.
Consistent with that, the only blocking signal that surfaced at all was blocking
predicting an org change at AAA (AUC 0.54) — real, but weak.

What this canNOT see, and what would make a better test: 40-man roster status,
option years remaining, service time, and contract control. Those are the actual
mechanism of blocking, and none of them are in this database. The negative
result is about depth-chart proxies, not about the idea itself.

Usage::

    from prospects.features.depth import resolve_franchise, depth_features
    s = resolve_franchise()      # season_stats + franchise / pos / lvl_rank
    d = depth_features(s)        # one row per affiliated MiLB player-season
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

from prospects import config

# MLB org codes -> canonical franchise. Relocations and renames collapse to a
# single franchise (OAK==ATH, FLA==MIA, Expos==Nationals, Indians==Guardians) so
# a depth chart stays continuous across the change.
MLB_CODE: dict[str, str] = {
    "ATH": "Athletics", "OAK": "Athletics", "ATL": "Braves", "AZ": "Diamondbacks",
    "BAL": "Orioles", "BOS": "Red Sox", "CHC": "Cubs", "CIN": "Reds",
    "CLE": "Guardians", "COL": "Rockies", "CWS": "White Sox", "DET": "Tigers",
    "FLA": "Marlins", "MIA": "Marlins", "HOU": "Astros", "KC": "Royals",
    "LAA": "Angels", "LAD": "Dodgers", "MIL": "Brewers", "MIN": "Twins",
    "NYM": "Mets", "NYY": "Yankees", "PHI": "Phillies", "PIT": "Pirates",
    "SD": "Padres", "SEA": "Mariners", "SF": "Giants", "STL": "Cardinals",
    "TB": "Rays", "TEX": "Rangers", "TOR": "Blue Jays", "WSH": "Nationals",
}

# Parent-club full names (as they appear in the affiliate map) -> franchise.
_NAME_TAIL: dict[str, str] = {
    "Athletics": "Athletics", "Braves": "Braves", "Diamondbacks": "Diamondbacks",
    "Orioles": "Orioles", "Red Sox": "Red Sox", "Cubs": "Cubs", "Reds": "Reds",
    "Guardians": "Guardians", "Indians": "Guardians", "Rockies": "Rockies",
    "White Sox": "White Sox", "Tigers": "Tigers", "Marlins": "Marlins",
    "Astros": "Astros", "Royals": "Royals", "Angels": "Angels",
    "Dodgers": "Dodgers", "Brewers": "Brewers", "Twins": "Twins", "Mets": "Mets",
    "Yankees": "Yankees", "Phillies": "Phillies", "Pirates": "Pirates",
    "Padres": "Padres", "Mariners": "Mariners", "Giants": "Giants",
    "Cardinals": "Cardinals", "Rays": "Rays", "Devil Rays": "Rays",
    "Rangers": "Rangers", "Blue Jays": "Blue Jays", "Nationals": "Nationals",
    "Expos": "Nationals",
}

POS_GROUP: dict[str, str] = {
    "P": "P", "RHP": "P", "LHP": "P", "SP": "P", "RP": "P", "TWP": "P",
    "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
    "OF": "OF", "CF": "OF", "LF": "OF", "RF": "OF", "DH": "DH", "IF": "IF",
}

LEVEL_RANK: dict[str, int] = {"RK": 0, "A-": 1, "A": 2, "A+": 3,
                              "AA": 4, "AAA": 5, "MLB": 6}
MLB_RANK = LEVEL_RANK["MLB"]


def _franchise_from_name(s) -> str | None:
    if not isinstance(s, str):
        return None
    for tail, canon in _NAME_TAIL.items():
        if s.endswith(tail):
            return canon
    return None


def pos_group(p) -> str | None:
    """Normalise a position string to a depth-chart group.

    Deliberately coarse, with two lossy merges worth knowing about: OF collapses
    LF/CF/RF, and P collapses SP/RP — the latter matters, because starter and
    reliever paths to the majors are not the same path.
    """
    if not isinstance(p, str):
        return None
    p = p.strip().upper()
    if p in POS_GROUP:
        return POS_GROUP[p]
    return POS_GROUP.get(re.split(r"[/\-,]", p)[0].strip())


def resolve_franchise(db: str | None = None, y0: int = 2005,
                      y1: int = 2026) -> pd.DataFrame:
    """season_stats rows plus point-in-time ``franchise``, ``pos``, ``lvl_rank``.

    Rows that cannot be attributed to an MLB franchise (Mexican League, NCAA,
    NULL org) come back with ``franchise`` NaN. Drop them — they are not part of
    any organisation's depth chart.
    """
    db = db or str(config.model_db())
    con = sqlite3.connect(db)
    s = pd.read_sql(
        f"""SELECT player_id, season_year, level, org, primary_position,
                   age_during_season, pa, ip, pct_woba, pct_fip
            FROM season_stats WHERE season_year BETWEEN {y0} AND {y1}""", con)
    con.close()

    m = pd.read_csv(config.REPO_ROOT / "reference" / "baseballcube"
                    / "affiliate_org_map.csv")
    m = (m[(m.season >= y0) & (m.season <= y1)][["season", "abbrev", "parent_org"]]
         .drop_duplicates(["season", "abbrev"]))
    m["fr_milb"] = m.parent_org.map(_franchise_from_name)

    s = s.merge(m[["season", "abbrev", "fr_milb"]],
                left_on=["season_year", "org"], right_on=["season", "abbrev"],
                how="left").drop(columns=["season", "abbrev"])
    is_mlb = s.level.eq("MLB")
    # MLB codes bypass the affiliate map — several collide with historical clubs
    # in it, e.g. 'COL' -> "Columbus Buckeyes".
    s["franchise"] = s.fr_milb.where(~is_mlb, s.org.map(MLB_CODE))
    s["lvl_rank"] = s.level.map(LEVEL_RANK)
    s["pos"] = s.primary_position.map(pos_group)
    # pct_* are 0-1 with higher = better rate stat. FIP is inverted (low is
    # good), so a high pct_fip is a bad pitcher — flip it.
    s["perf_pct"] = np.where(s.pos.eq("P"), 1.0 - s.pct_fip, s.pct_woba)
    return s.drop(columns=["fr_milb"])


def org_change_rate(s: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year rate at which players change franchise, by level.

    The headline number for interpreting any org-context feature: AAA runs ~39%,
    so org attachment there is transient.
    """
    s = s[s.franchise.notna()].sort_values(["player_id", "season_year"])
    nx = s.assign(next_fr=s.groupby("player_id").franchise.shift(-1),
                  next_yr=s.groupby("player_id").season_year.shift(-1))
    nx = nx[nx.next_yr == nx.season_year + 1].copy()
    nx["changed_org"] = (nx.next_fr != nx.franchise).astype(int)
    return (nx.groupby("level").changed_org.agg(n="size", rate="mean")
              .sort_values("rate"))


def depth_features(s: pd.DataFrame, min_pa: int = 50,
                   min_ip: int = 20) -> pd.DataFrame:
    """One row per affiliated MiLB player-season with blocking/opportunity cols.

    See the module docstring: these are validated but do NOT predict debut or
    establishment. Kept because the franchise resolution underneath them is
    sound and the depth chart itself is useful for explanation and display.
    """
    s = s[s.franchise.notna() & s.pos.notna() & s.lvl_rank.notna()].copy()
    s["lvl_rank"] = s.lvl_rank.astype(int)
    s["is_real"] = ((s.pa.fillna(0) >= min_pa)
                    | (s.ip.fillna(0) >= min_ip)).astype(int)
    real = s[s.is_real.eq(1)]

    mlb = real[real.level.eq("MLB")]
    inc = (mlb.groupby(["franchise", "season_year", "pos"])
              .agg(n_mlb_inc=("player_id", "nunique"),
                   inc_best_pct=("perf_pct", "max"),
                   inc_med_pct=("perf_pct", "median"),
                   inc_min_age=("age_during_season", "min"))
              .reset_index())

    tot = (real.groupby(["franchise", "season_year", "pos", "lvl_rank"])
               .player_id.nunique().rename("n").reset_index())
    piv = tot.pivot_table(index=["franchise", "season_year", "pos"],
                          columns="lvl_rank", values="n", fill_value=0)
    for r in range(MLB_RANK + 1):
        if r not in piv.columns:
            piv[r] = 0
    piv = piv[sorted(piv.columns)]
    arr = piv.values
    above = piv.reset_index()[["franchise", "season_year", "pos"]].copy()
    above_cols, at_cols = [], []
    for r in range(MLB_RANK + 1):
        above[f"_above_{r}"] = arr[:, r + 1:].sum(axis=1) if r < MLB_RANK else 0
        above[f"_at_{r}"] = arr[:, r]
        above_cols.append(f"_above_{r}")
        at_cols.append(f"_at_{r}")

    p = real[real.level.ne("MLB")].copy()
    p = p.merge(inc, on=["franchise", "season_year", "pos"], how="left")
    p = p.merge(above, on=["franchise", "season_year", "pos"], how="left")

    # Pick, per row, the column matching that row's own level.
    ix = p.lvl_rank.to_numpy().astype(int)
    rows = np.arange(len(p))
    p["n_above_pos"] = p[above_cols].to_numpy()[rows, ix]
    p["n_same_level_pos"] = p[at_cols].to_numpy()[rows, ix]
    p = p.drop(columns=above_cols + at_cols)

    p["n_mlb_inc"] = p.n_mlb_inc.fillna(0)
    p["inc_min_age"] = p.inc_min_age.where(p.inc_min_age.between(17, 50))
    p["levels_to_mlb"] = MLB_RANK - p.lvl_rank
    p["org_pos_rank"] = (p.groupby(["franchise", "season_year", "pos"])
                          .perf_pct.rank(ascending=False, method="min"))
    p["org_pos_n"] = (p.groupby(["franchise", "season_year", "pos"])
                       .perf_pct.transform("size"))
    p["org_pos_pctile"] = 1 - (p.org_pos_rank - 1) / p.org_pos_n.clip(lower=1)
    # A young AND good incumbent sitting at your position — the sharpest
    # available proxy for a genuinely hard block.
    p["hard_block"] = ((p.inc_min_age < 27) & (p.inc_best_pct > 0.75)).astype(float)
    p.loc[p.inc_min_age.isna() | p.inc_best_pct.isna(), "hard_block"] = np.nan

    # Position-normalise the raw counts. Without this they mostly encode
    # position (an org rosters ~20 MLB pitchers and ~3 catchers), not blocking.
    for f in ("n_mlb_inc", "n_above_pos", "n_same_level_pos", "inc_best_pct",
              "inc_min_age"):
        g = p.groupby(["pos", "season_year"])[f]
        p[f"{f}_z"] = ((p[f] - g.transform("mean"))
                       / g.transform("std").replace(0, np.nan))
    return p
