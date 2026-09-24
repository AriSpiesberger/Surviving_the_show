"""Equal-weight combinations of the saved v3-all component trajectories (2026-09-23).

Candidates are fixed in advance and deliberately few and equal-weighted (no weights fit on val),
because they are chosen on the same held-out val they are scored on.

    python -m prospects.model.train.exp_combo
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import prep_base, realized_by_h

_RUN = config.run()
DB = str(config.model_db())
CANDS = {
    "MLB_DEBUT": [("control", "v3"), ("control", "v3", "haz")],
    "TOP_100_PROSPECT": [("control", "v3"), ("control", "haz"), ("control", "v3", "haz")],
    "ESTABLISHED_MLB": [("control", "v3cond"), ("control", "v3"), ("v3cond", "haz"), ("control", "v3cond", "haz"),
                        ("control", "v3", "v3cond", "haz")],
    "STAR_PLUS_ELITE": [("v3cond",), ("haz",), ("control", "v3cond"), ("v3cond", "haz"),
                        ("control", "v3cond", "haz"), ("v3", "v3cond", "haz")],
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["exp_v3_all", "exp_v3_all_hazfull"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    zs = [np.load(REPO_ROOT / "runs" / d / "val_preds.npz", allow_pickle=True) for d in args.dirs]
    val = prep_base(pd.read_csv(_RUN.oof_val_long, low_memory=False), DB, max_entry=2020)
    assert all((val.player_id.to_numpy() == z["pid"]).all() for z in zs)
    comp = {}
    for z in zs:
        for k in z.files:
            if "__" in k:
                ev, a = k.split("__", 1)
                comp.setdefault(ev, {}).setdefault(a, z[k])
    yf, vpid = val.years_fwd.to_numpy(), val.player_id.to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for ev, cands in CANDS.items():
        el = (val[f"eligible_{ev}"] == 1).to_numpy()
        for h in (3, 6):
            mm = (yf >= h) & el
            yy = np.asarray(realized_by_h(val[mm], ev, h), dtype=float)
            pl = vpid[mm]
            upl = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in upl}
            draws = [np.concatenate([pos[q] for q in rng.choice(upl, len(upl), replace=True)]) for _ in range(300)]
            base = comp[ev]["control"][mm, h - 1]
            for c in cands:
                p = np.mean([comp[ev][a][mm, h - 1] for a in c], axis=0)
                d = np.array([average_precision_score(yy[ix], p[ix]) - average_precision_score(yy[ix], base[ix])
                              for ix in draws if yy[ix].sum()])
                rows.append({"event": ev, "h": h, "combo": "+".join(c), "ap": average_precision_score(yy, p),
                             "d_ap": d.mean(), "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5),
                             "calib": p.mean() / yy.mean()})
    res = pd.DataFrame(rows)
    out = REPO_ROOT / "runs" / f"exp_combo{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "combos.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
