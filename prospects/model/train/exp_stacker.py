"""Learned stacker vs the fixed 50/50 mix (architecture B, 2026-09-23).

exp_v3_all showed that 0.5 * joint + 0.5 * v3 beats either model on every event. A fixed weight
cannot know that the encoder may matter more early in a career and the joint's hazard stack
more later. The stacker learns the combination per event and horizon:

    logit P = b0 + sum_m b_m * logit(P_m) + sum_m c_m * logit(P_m) * yip_c + d * yip_c

over the calibrated component trajectories exp_v3_all saved for the held-out val players
(control = deployed joint, v3, v3cond where present). It is fit and scored with 2-fold
cross-fitting BY PLAYER inside val — each half is scored by a stacker that never saw it — so
the comparison with the fixed mix on the same rows is honest. Only ~6 coefficients per
(event, h), so the half-sample fits are stable.

    python -m prospects.model.train.exp_stacker --preds runs/exp_v3_all/val_preds.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import prep_base, realized_by_h

_RUN = config.run()
DB = str(config.model_db())
EVAL_H = (3, 6)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", nargs="+", default=[str(REPO_ROOT / "runs" / "exp_v3_all" / "val_preds.npz")],
                    help="one or more exp_v3_all val_preds.npz; components are merged by name "
                         "(the control must be identical across files)")
    ap.add_argument("--components", nargs="*", default=["control", "v3", "v3cond", "haz"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    zs = [np.load(f, allow_pickle=True) for f in args.preds]
    z = zs[0]
    for other in zs[1:]:
        assert (other["pid"] == z["pid"]).all(), "prediction files cover different val rows"
    val = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    assert (val["player_id"].to_numpy() == z["pid"]).all(), "val rows do not line up with saved predictions"
    yip = z["yip"].astype(float)
    yf = val["years_fwd"].to_numpy()
    vpid = val["player_id"].to_numpy()
    upl = np.unique(vpid)
    half = np.isin(vpid, np.random.default_rng(11).choice(upl, upl.size // 2, replace=False))
    comps = {}
    for zz in zs:
        for key in zz.files:
            if "__" in key:
                ev, arm = key.split("__", 1)
                comps.setdefault(ev, {}).setdefault(arm, zz[key])
    rows, boots = [], []
    rng = np.random.default_rng(0)
    for ev, arms in comps.items():
        base = [a for a in args.components if a in arms]
        el = (val[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in val.columns else np.ones(len(val), bool)
        for h in EVAL_H:
            mm = (yf >= h) & el
            yy = np.asarray(realized_by_h(val[mm], ev, h), dtype=int)
            if yy.sum() < 10:
                continue
            yc = (yip[mm] - 2.0) / 2.0
            L = np.column_stack([logit(arms[a][mm, h - 1]) for a in base])
            F = np.column_stack([L, L * yc[:, None], yc])
            hm = half[mm]
            stk = np.full(len(yy), np.nan)
            for side in (True, False):
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(F[hm == side], yy[hm == side])
                stk[hm != side] = lr.predict_proba(F[hm != side])[:, 1]
            cand = {"control": arms["control"][mm, h - 1], "stacker": stk}
            for mix in ("mix_v3", "mix_cond", "mix_haz"):
                if mix in arms:
                    cand[mix] = arms[mix][mm, h - 1]
            for name, p in cand.items():
                rows.append({"event": ev, "h": h, "arm": name, "pos": int(yy.sum()),
                             "ap": average_precision_score(yy, p), "auc": roc_auc_score(yy, p),
                             "calib": p.mean() / yy.mean()})
            pl = vpid[mm]
            upp = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in upp}
            draws = [np.concatenate([pos[q] for q in rng.choice(upp, len(upp), replace=True)])
                     for _ in range(args.n_boot)]
            best_mix = max((m for m in cand if m.startswith("mix")),
                           key=lambda m: average_precision_score(yy, cand[m]), default="control")
            for ref in ("control", best_mix):
                d = np.array([average_precision_score(yy[ix], stk[ix]) - average_precision_score(yy[ix], cand[ref][ix])
                              for ix in draws if yy[ix].sum()])
                boots.append({"event": ev, "h": h, "vs": ref, "d_ap": d.mean(),
                              "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
    res, bt = pd.DataFrame(rows), pd.DataFrame(boots)
    out = REPO_ROOT / "runs" / f"exp_stacker{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "metrics.csv", index=False)
    bt.to_csv(out / "bootstrap.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print("\n===== AP(stacker) - AP(reference), paired player bootstrap =====")
    print(bt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
