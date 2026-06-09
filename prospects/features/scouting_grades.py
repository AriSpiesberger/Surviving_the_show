"""Point-in-time scouting-grade features (FanGraphs Board + TWTC).

Loads scratch/fangraphs_board/scouting_grades_pointintime.csv (built by
scripts/build_scouting_grades.py) and exposes a no-lookahead lookup: for a
player at as_of_year S, return the most recent grade snapshot with season <= S.
Overlap seasons prefer fg_board over twtc.

Used by prospects.features.scouting to append these as model features, so they
flow into the panel / hazards / scoring uniformly. Missing -> NaN (HistGB-safe).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_CSV = (Path(__file__).resolve().parents[2] / "scratch" / "fangraphs_board"
        / "scouting_grades_pointintime.csv")

# Set SCOUTING_GRADES_OFF=1 to disable these features (baseline A/B run).
_OFF = os.environ.get("SCOUTING_GRADES_OFF") == "1"

if _CSV.exists() and not _OFF:
    _df = pd.read_csv(_CSV, low_memory=False)
    _FEAT_COLS = [c for c in _df.columns
                  if c not in ("player_id", "season", "source")]
    # one row per (player, season); prefer fg_board over twtc on overlap
    _df = (_df.sort_values(["player_id", "season", "source"])
           .drop_duplicates(["player_id", "season"], keep="first"))
    SCOUTING_GRADE_NAMES = [f"scout_{c}" for c in _FEAT_COLS]
    _BY_PLAYER: dict[str, tuple] = {}
    for pid, g in _df.groupby("player_id"):
        g = g.sort_values("season")
        _BY_PLAYER[str(pid)] = (g["season"].to_numpy(),
                                g[_FEAT_COLS].to_numpy(dtype=float))
else:  # data not built yet -> no scouting features (graceful, deterministic)
    SCOUTING_GRADE_NAMES = []
    _BY_PLAYER = {}

_NAN = {n: np.nan for n in SCOUTING_GRADE_NAMES}


def scouting_grade_dict(player_id, as_of_year) -> dict[str, float]:
    """Return {scout_*: value} for the latest snapshot with season <= as_of_year
    (no lookahead). Always returns all SCOUTING_GRADE_NAMES; NaN where absent."""
    e = _BY_PLAYER.get(str(player_id))
    if e is None or as_of_year is None:
        return dict(_NAN)
    seasons, vals = e
    i = int(np.searchsorted(seasons, as_of_year, side="right")) - 1
    if i < 0:
        return dict(_NAN)
    return {n: v for n, v in zip(SCOUTING_GRADE_NAMES, vals[i])}


# ---------------------------------------------------------------------------
# Compact CURRENT-snapshot summary for the joint XGB (which otherwise only sees
# hazard probs). Point-in-time via backward merge_asof; 0-filled so the XGB's
# StandardScaler stays happy, with scout_is_scouted as the presence gate.
# ---------------------------------------------------------------------------
SCOUTING_SUMMARY_COLS = ["scout_fv", "scout_ovr_rank", "scout_eta_gap",
                         "scout_risk", "scout_is_scouted"]
_SUMMARY_TBL = None


def _summary_table():
    global _SUMMARY_TBL
    if _SUMMARY_TBL is None:
        if _OFF or not _CSV.exists():
            _SUMMARY_TBL = pd.DataFrame(
                columns=["player_id", "season", "fv", "ovr_rank", "eta", "risk"])
        else:
            s = pd.read_csv(_CSV, low_memory=False)
            s = (s.sort_values(["player_id", "season", "source"])
                 .drop_duplicates(["player_id", "season"], keep="first"))
            _SUMMARY_TBL = (s[["player_id", "season", "fv", "ovr_rank", "eta", "risk"]]
                            .sort_values("season").reset_index(drop=True))
    return _SUMMARY_TBL


def attach_scouting_summary(df):
    """Add SCOUTING_SUMMARY_COLS to a long df (needs player_id, snap_year) via
    point-in-time backward merge_asof (latest grade with season <= snap_year)."""
    s = _summary_table()
    if s.empty or "snap_year" not in df.columns:
        df = df.copy()
        for c in SCOUTING_SUMMARY_COLS:
            df[c] = 0.0
        return df
    m = pd.merge_asof(df.sort_values("snap_year"), s, left_on="snap_year",
                      right_on="season", by="player_id", direction="backward")
    m["scout_is_scouted"] = m["season"].notna().astype(float)
    m["scout_fv"] = m["fv"].fillna(0.0)
    m["scout_ovr_rank"] = m["ovr_rank"].fillna(9999.0)   # unranked = worst
    m["scout_risk"] = m["risk"].fillna(0.0)
    m["scout_eta_gap"] = (m["eta"] - m["snap_year"]).fillna(0.0)
    return m.drop(columns=["fv", "ovr_rank", "eta", "risk", "season"],
                  errors="ignore")
