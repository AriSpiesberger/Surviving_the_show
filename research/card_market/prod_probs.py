"""Production (v3.5) calibrated probabilities for every event at every horizon h1..h6, snap 2026.

The buy list publishes debut at several horizons but the ceiling events only at h6; the EV needs
P(TOP_100 within 1 year) next to P(debut within 1 year). Same scoring path as
prospects.buylist.build (score_trajectory + the v3.5 calibrators), nothing refit.

    python prod_probs.py   ->  raw/prod_probs_2026.csv
"""
from __future__ import annotations

import sys

import pandas as pd

from load import HERE, REPO

sys.path.insert(0, str(REPO))
from prospects.buylist.build import _add_feats, _join_current_level, _join_prospect_meta  # noqa: E402
from prospects.model.joint2 import load_calibrators, make_cal_fn, score_trajectory  # noqa: E402

MODELS = REPO / "runs" / "current" / "models"
LONG = REPO / "runs" / "current" / "scored" / "snap2026_long.csv"
DBP = str(REPO / "prospects_snapshot.db")
EVENTS = ["MLB_DEBUT", "TOP_100_PROSPECT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]


def main():
    df = pd.read_csv(LONG)
    df = _add_feats(df, DBP)
    df = _join_prospect_meta(df, DBP)
    df = _join_current_level(df, DBP)
    scored, _ = score_trajectory(str(MODELS / "v3.5.pkl"), df, DBP)
    cal_fn = make_cal_fn(load_calibrators(str(MODELS / "calibrators_v3.5.pkl")), scored)
    out = df[["player_id", "cur_level_2026", "year_top_100"]].copy()
    for ev in EVENTS:
        for h in range(1, 7):
            col = f"xp_{ev}_h{h}"
            if col in scored.columns:
                out[f"p_{ev}_{h}y"] = cal_fn(scored[col].to_numpy(), ev, h)
    out.to_csv(HERE / "raw" / "prod_probs_2026.csv", index=False)
    print(out.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
