"""Create synthetic "twin" players by resampling real careers at the season-stat level (2026-09-24).

User idea: augment by creating more players — draw samples from each player's underlying
random process and add missingness. A twin is the same talent with different luck and gaps:

  hitters   per season, outcome shares (1B, 2B, 3B, HR, BB, HBP, K, SF, other outs) ~ Dirichlet(
            observed counts + PRIOR x level-average shares); new counts ~ Multinomial(PA, shares);
            AVG / OBP / SLG / ISO / K% / BB% / BABIP recomputed, wOBA re-derived by the production
            woba backfill
  pitchers  per season, batters faced split into (K, BB, HBP, HR, non-HR hits, outs) the same way;
            K9 / BB9 / HR9 / WHIP recomputed, ERA scaled by a run-value ratio, FIP re-derived
  missing   each earlier MiLB season is dropped with prob --drop; a block of detail columns
            (batted-ball / plate-discipline) is blanked with prob --blank
  labels    the twin keeps the real player's outcomes, draft record and rankings (a twin is an
            alternative realisation of the same future)

Percentiles for twin rows are ranked against the REAL (level, season) cohort only, so twins never
move any real player's features. Twins get ids tw<k>_<player_id> and no mlbam_id. Only the chosen
fit players are twinned (never val). Writes prospects_aug.db (a copy of the model DB) and
runs/experiments/aug_twins/twins_long.csv (the twins' landmark rows = the originals' rows re-keyed).

    python tools/build_twins.py --frac 0.25 --k 2
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H_CATS = ["s1", "d", "t", "hr", "bb", "hbp", "so", "sf", "out"]
P_CATS = ["so", "bb", "hbp", "hr", "h", "out"]
DETAIL = ["ground_outs", "air_outs", "fly_outs", "line_outs", "pop_outs", "ground_hits", "fly_hits",
          "line_hits", "pop_hits", "balls_in_play", "pitches_seen", "total_swings", "swings_and_misses",
          "p_pitches", "p_strikes", "p_total_swings", "p_swings_and_misses", "p_ground_outs", "p_air_outs",
          "p_fly_outs", "p_line_outs", "p_pop_outs", "p_ground_hits", "p_fly_hits", "p_line_hits",
          "p_pop_hits", "p_balls_in_play"]


def nz(v):
    return 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def hit_counts(r):
    """Outcome counts built from the COUNTS alone. 2.6% of rows carry a PA total that covers more
    than the counts do (e.g. PA from two teams, counts from one: 306 PA vs 123 AB), so the
    category total is the counts' own PA, never the stored PA."""
    if r["hits"] is None or r["so"] is None or r["ab"] is None or nz(r["ab"]) <= 0:
        return None
    s1 = nz(r["hits"]) - nz(r["doubles"]) - nz(r["triples"]) - nz(r["home_runs"])
    out = nz(r["ab"]) - nz(r["hits"]) - nz(r["so"]) + nz(r["sac_bunts"])
    c = np.array([s1, nz(r["doubles"]), nz(r["triples"]), nz(r["home_runs"]), nz(r["bb"]), nz(r["hbp"]),
                  nz(r["so"]), nz(r["sf"]), out])
    return c if (c >= 0).all() and c.sum() > 0 else None


def pit_counts(r):
    """Batters-faced categories from the counts: outs in play = recorded outs - strikeouts."""
    if r["p_so"] is None or r["p_hits"] is None or r["p_outs"] is None:
        return None
    c = np.array([nz(r["p_so"]), nz(r["p_bb"]), nz(r["p_hbp"]), nz(r["p_hr"]),
                  nz(r["p_hits"]) - nz(r["p_hr"]), nz(r["p_outs"]) - nz(r["p_so"])])
    return c if (c >= 0).all() and c.sum() > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(ROOT / "prospects_snapshot.db"))
    ap.add_argument("--out", default=str(ROOT / "prospects_aug.db"))
    ap.add_argument("--frac", type=float, default=0.25, help="fit players to twin (same sample as the quick screen)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--prior", type=float, default=50.0, help="Dirichlet prior strength (PA / BF)")
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--blank", type=float, default=0.10)
    a = ap.parse_args()
    from prospects import config
    from prospects.features.percentiles import HIT_METRICS, PIT_METRICS, QUAL_IP, QUAL_PA, MIN_COHORT, \
        _memoize_advanced, _percentiles

    run = config.run("current")
    fit_long = pd.read_csv(run.oof_stacked_long, low_memory=False)
    u = np.sort(fit_long.player_id.unique())
    keep = set(np.random.default_rng(a.seed).choice(u, int(len(u) * a.frac), replace=False)) if a.frac < 1 else set(u)
    shutil.copy2(a.src, a.out)
    con = sqlite3.connect(a.out)
    con.row_factory = sqlite3.Row
    rng = np.random.default_rng(a.seed + 1000)

    ss = [dict(r) for r in con.execute("SELECT * FROM season_stats")]
    by_pid = {}
    for r in ss:
        by_pid.setdefault(r["player_id"], []).append(r)
    # level-average outcome shares (the Dirichlet prior)
    hsum, psum = {}, {}
    for r in ss:
        if r["level"] == "MLB":
            continue
        h, p = hit_counts(r), pit_counts(r)
        if h is not None:
            hsum[r["level"]] = hsum.get(r["level"], 0) + h
        if p is not None:
            psum[r["level"]] = psum.get(r["level"], 0) + p
    hprior = {k: v / v.sum() for k, v in hsum.items()}
    pprior = {k: v / v.sum() for k, v in psum.items()}

    cols = list(ss[0].keys())
    twins_ss, twin_ids = [], []
    for pid in sorted(keep):
        rows = by_pid.get(pid, [])
        milb = sorted((r for r in rows if r["level"] != "MLB"), key=lambda r: r["season_year"])
        last_year = milb[-1]["season_year"] if milb else None
        for k in range(1, a.k + 1):
            tid = f"tw{k}_{pid}"
            twin_ids.append((tid, pid))
            for r in rows:
                t = dict(r, player_id=tid)
                if r["level"] != "MLB":
                    if r["season_year"] != last_year and rng.random() < a.drop:
                        continue                                    # a missing season
                    h = hit_counts(r)
                    if h is not None:
                        pr = hprior.get(r["level"], h / h.sum())
                        sh = rng.dirichlet(h + a.prior * pr + 1e-3)
                        n = rng.multinomial(int(round(h.sum())), sh).astype(float)
                        s1, d, tr, hr, bb, hbp, so, sf, out = n
                        hits = s1 + d + tr + hr
                        ab = max(s1 + d + tr + hr + so + out - nz(r["sac_bunts"]), 1.0)
                        pa_c = max(n.sum(), 1.0)
                        tb = s1 + 2 * d + 3 * tr + 4 * hr
                        t.update(hits=hits, doubles=d, triples=tr, home_runs=hr, bb=bb, hbp=hbp, so=so, sf=sf,
                                 ab=ab, total_bases=tb, ibb=round(nz(r["ibb"]) * (bb / max(nz(r["bb"]), 1))),
                                 avg=hits / ab, obp=(hits + bb + hbp) / max(ab + bb + hbp + sf, 1),
                                 slg=tb / ab, iso=(tb - hits) / ab, k_pct=so / pa_c,
                                 bb_pct=bb / pa_c,
                                 babip=(hits - hr) / max(ab - so - hr + sf, 1), woba=None)
                    p = pit_counts(r)
                    if p is not None and nz(r["ip"]) > 0:
                        pr = pprior.get(r["level"], p / p.sum())
                        sh = rng.dirichlet(p + a.prior * pr + 1e-3)
                        so, bb, hbp, hr, h1, out = rng.multinomial(int(round(p.sum())), sh).astype(float)
                        ip = nz(r["ip"])
                        rv = lambda H, HR, W: 0.5 * H + 1.44 * HR + 0.33 * W
                        old = rv(p[4], p[3], p[1] + p[2])
                        ratio = rv(h1, hr, bb + hbp) / old if old > 0 else 1.0
                        t.update(p_so=so, p_bb=bb, p_hbp=hbp, p_hr=hr, p_hits=h1 + hr,
                                 k9=9 * so / ip, bb9=9 * bb / ip, hr9=9 * hr / ip, whip=(bb + h1 + hr) / ip,
                                 era=(nz(r["era"]) * ratio) if r["era"] is not None else None,
                                 p_earned_runs=nz(r["p_earned_runs"]) * ratio, fip=None)
                    if rng.random() < a.blank:
                        for c in DETAIL:
                            t[c] = None                           # a missing detail block
                    for c in cols:
                        if c.startswith("pct_"):
                            t[c] = None
                twins_ss.append(t)
    print(f"[twins] {len(keep):,} fit players x {a.k} twins = {len(twin_ids):,} twins; {len(twins_ss):,} season rows")

    qcols = ",".join(cols)
    with con:
        con.executemany(f"INSERT INTO season_stats ({qcols}) VALUES ({','.join('?' * len(cols))})",
                        [tuple(t[c] for c in cols) for t in twins_ss])
        for tbl, extra in (("prospects", {"mlbam_id": None}), ("career_outcomes", {}), ("rankings_history", {})):
            tc = [r[1] for r in con.execute(f"PRAGMA table_info({tbl})")]
            src = pd.read_sql(f"SELECT * FROM {tbl}", con)
            src = src[src.player_id.isin(keep)]
            out = []
            for tid, pid in twin_ids:
                s = src[src.player_id == pid]
                for _, row in s.iterrows():
                    rr = row.to_dict()
                    rr.update(player_id=tid, **extra)
                    out.append(tuple(rr[c] for c in tc))
            con.executemany(f"INSERT INTO {tbl} ({','.join(tc)}) VALUES ({','.join('?' * len(tc))})", out)
    con.close()
    from prospects.data.backfills import woba_backfill
    woba_backfill.backfill(a.out, verbose=False)                       # twins' wOBA / FIP from their counts

    # percentiles of twin rows against the REAL cohort only
    con = sqlite3.connect(a.out)
    con.row_factory = sqlite3.Row
    allr = [dict(r) for r in con.execute("SELECT * FROM season_stats")]
    _memoize_advanced(allr)
    real = [r for r in allr if not r["player_id"].startswith("tw")]
    tw = [r for r in allr if r["player_id"].startswith("tw")]
    coh = {}
    for r in real:
        coh.setdefault((r["level"], r["season_year"]), []).append(r)
    upd = {}
    for metrics, qk, qm in ((HIT_METRICS, "pa", QUAL_PA), (PIT_METRICS, "ip", QUAL_IP)):
        refs = {}
        for key, g in coh.items():
            q = [r for r in g if (r.get(qk) or 0) >= qm]
            if len(q) < MIN_COHORT:
                continue
            for name, ex in metrics:
                ref = np.array(sorted(v for v in (ex(r) for r in q) if v is not None and np.isfinite(v)))
                if len(ref) >= MIN_COHORT:
                    refs[(key, name)] = ref
        for r in tw:
            key = (r["level"], r["season_year"])
            for name, ex in metrics:
                ref = refs.get((key, name))
                v = ex(r)
                if ref is not None and v is not None and np.isfinite(v):
                    upd.setdefault((r["player_id"], r["season_year"], r["level"]), {})[f"pct_{name}"] = \
                        float(_percentiles(ref, np.array([v]))[0])
    with con:
        for (pid, y, lv), vals in upd.items():
            con.execute(f"UPDATE season_stats SET {', '.join(k + ' = ?' for k in vals)} "
                        "WHERE player_id = ? AND season_year = ? AND level = ?", (*vals.values(), pid, y, lv))
    con.close()
    print(f"[twins] percentiles for {len(upd):,} twin rows ranked against the real cohorts")

    # twins' landmark rows: the originals' fit-long rows re-keyed
    tl = []
    for k in range(1, a.k + 1):
        t = fit_long[fit_long.player_id.isin(keep)].copy()
        t["player_id"] = f"tw{k}_" + t["player_id"]
        tl.append(t)
    dest = ROOT / "runs" / "experiments" / "aug_twins"
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(tl, ignore_index=True).to_csv(dest / "twins_long.csv", index=False)
    print(f"[twins] wrote {a.out} and {dest / 'twins_long.csv'}")


if __name__ == "__main__":
    main()
