"""Spec-blend comparison across exp_v3_all runs (2026-09-28).

Rebuilds the production per-event component blend (prospects.model.v3.DEFAULT_SPEC, minus nhaz,
which the harness does not train) from each run's val_preds.npz and scores variants against the
baseline blend with a paired player bootstrap on the identical val rows.

    python -m prospects.model.train.spec_compare --base runs/experiments/exp_creative_q25 \
        --variant "rk:MLB_DEBUT=control,v3,rk" --variant "pre96:@runs/experiments/exp_pre96fix_q25"

A variant is NAME:EVENT=arm,arm;EVENT=arm (arms from the base run; unspecified events keep the
baseline spec) or NAME:@RUN_DIR (that run's own baseline spec).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps

from prospects import config
from prospects.model.joint import realized_by_h

SPEC = {"MLB_DEBUT": ["control", "v3"], "TOP_100_PROSPECT": ["control", "haz"],
        "ESTABLISHED_MLB": ["control", "v3", "v3cond", "haz"], "STAR_PLUS_ELITE": ["v3cond", "haz"]}
from prospects.evaluation.run import EVENT_WEIGHTS as WEIGHT   # debut x2, as in the headline weighted AP


def load(run):
    z = np.load(f"{run}/val_preds.npz", allow_pickle=True)
    k = pd.DataFrame({"pid": z["pid"], "snap": z["snap_year"].astype(int)})
    return z, k


def blend(z, spec):
    return {ev: np.mean([z[f"{ev}__{a}"] for a in arms], axis=0) for ev, arms in spec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--val", default=str(config.run().oof_val_long))
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args()

    zb, kb = load(a.base)
    v = pd.read_csv(a.val, low_memory=False)
    cols = ["player_id", "snap_year", "years_fwd"] + [c for c in v.columns if c.startswith(("trigger_", "eligible_"))]
    v = v[cols].drop_duplicates(["player_id", "snap_year"]).rename(columns={"player_id": "pid", "snap_year": "snap"})
    lab = kb.merge(v, on=["pid", "snap"], how="left")
    arms = {"base": blend(zb, SPEC)}
    for spec in a.variant:
        name, body = spec.split(":", 1)
        if body.startswith("@"):
            zo, ko = load(body[1:])
            assert (ko.pid.to_numpy() == kb.pid.to_numpy()).all(), f"{name}: val rows differ"
            arms[name] = blend(zo, SPEC)
        else:
            s = dict(SPEC)
            for part in body.split(";"):
                ev, al = part.split("=")
                s[ev] = al.split(",")
            arms[name] = blend(zb, s)

    rng = np.random.default_rng(0)
    rows = []
    for ev in SPEC:
        el = (lab[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in lab else np.ones(len(lab), bool)
        for h in (3, 6):
            mm = (lab.years_fwd.to_numpy() >= h) & el
            yy = np.asarray(realized_by_h(lab[mm].rename(columns={"snap": "snap_year"}), ev, h), float)
            pl = lab.pid.to_numpy()[mm]
            u = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in u}
            draws = [np.concatenate([pos[q] for q in rng.choice(u, len(u))]) for _ in range(a.n_boot)]
            draws = [d for d in draws if yy[d].sum()]
            B = arms["base"][ev][mm, h - 1]
            for name, P in arms.items():
                p = P[ev][mm, h - 1]
                r = {"event": ev, "h": h, "arm": name, "pos": int(yy.sum()), "ap": aps(yy, p)}
                if name != "base":
                    d = np.array([aps(yy[x], p[x]) - aps(yy[x], B[x]) for x in draws])
                    r.update(d_ap=aps(yy, p) - aps(yy, B), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5))
                rows.append(r)
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    for h in (3, 6):
        w = df[df.h == h].pivot_table(index="arm", columns="event", values="ap")
        print(f"weighted AP h{h}:", (w.mul(pd.Series(WEIGHT)).sum(axis=1) / sum(WEIGHT.values())).round(4).to_dict())


if __name__ == "__main__":
    main()
