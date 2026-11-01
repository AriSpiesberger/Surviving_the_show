"""Short-horizon call-up model: who debuts in the next 3-30 days? (2026-10-05)

§1c of FINDINGS: card prices are flat until ~3 days before a debut and pop +70-85% over days
0-3, so buying late is free and the edge is knowing WHO is about to be called up. This builds a
weekly in-season panel from MLB transactions and fits a walk-forward model.

Panel: every Monday, April 7 -> September 22, seasons 2010-2025. A row is a not-yet-debuted
player whose current assignment (latest transaction to an affiliate, as of the day before the
snapshot) is AA or AAA. Rows whose debut is within 2 days are dropped: by then the price has
already moved, so a buyer could not have bought before the pop.

Features (all dated strictly before the snapshot):
  lvl_aaa         current assignment is AAA (else AA)
  days_at_lvl     days since that assignment;  days_season: days since April 1
  on40            on the 40-man (last 40-man event SE/OPT/CU/CLW vs OUT/REL/DFA)
  ever_opt        ever optioned (a 40-man player who has been up and down -- debuted players
                  are excluded, so this is mostly paper moves)
  org_il14_same   the MLB club's IL placements in the last 14 days at the player's position
                  group (pitcher / position player);  org_il14_all, org_il30_same
  org_callups14   the MLB club's selections + recalls in the last 14 days
  sept            snapshot in September (roster expansion)
  age, is_pitcher
  p_deb1          walk-forward P(debut next season) from light_model.py at the previous season
                  end (NaN when the player is not in the panel)
Target: debut in (t+2, t+30].

Model: HistGradientBoosting, walk-forward by season (season Y trained on seasons < Y).

    python callup.py build       # panel -> raw/callup_panel.parquet (people cache: raw/callup_people.jsonl)
    python callup.py fit         # walk-forward preds -> raw/callup_preds.csv + AP by season
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "raw"
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

SEASONS = range(2010, 2026)
LVL = {"AA": 5, "AAA": 6}
ADD40, DROP40 = {"SE", "OPT", "CU", "CLW"}, {"OUT", "REL", "DFA"}
PITCH = ("rhp", "lhp", "pitcher")
PANEL = RAW / "callup_panel.parquet"
PEOPLE = RAW / "callup_people.jsonl"


def team_maps():
    lv = json.loads((REPO / "reference" / "transactions" / "team_levels.json").read_text())
    par = pd.read_csv(REPO / "reference" / "transactions" / "team_parents.csv")
    parent = {(int(r.season), int(r.team_id)): (int(r.parent) if pd.notna(r.parent) else None) for r in par.itertuples()}
    return lv, parent


def people(ids):
    have = {}
    if PEOPLE.exists():
        for l in PEOPLE.open(encoding="utf-8"):
            d = json.loads(l)
            have[d["id"]] = d
    need = [i for i in ids if i not in have]
    with PEOPLE.open("a", encoding="utf-8") as f:
        for i in range(0, len(need), 150):
            url = "https://statsapi.mlb.com/api/v1/people?personIds=" + ",".join(map(str, need[i:i + 150]))
            for p in json.load(urllib.request.urlopen(url, timeout=60)).get("people", []):
                d = {k: p.get(k) for k in ("id", "fullName", "mlbDebutDate", "birthDate")}
                d["pos"] = (p.get("primaryPosition") or {}).get("abbreviation")
                f.write(json.dumps(d) + "\n")
                have[d["id"]] = d
            time.sleep(0.1)
    return pd.DataFrame(list(have.values()))


def build():
    from build_tx_features import load_tx
    tx = load_tx()
    tx = tx[tx.year.between(2009, 2025)]
    lv, parent = team_maps()
    tx["to_lvl"] = [lv.get(f"{y}:{int(t)}") if pd.notna(t) else None for y, t in zip(tx.year, tx.to_team)]
    tx["org"] = [parent.get((y, int(t))) if pd.notna(t) else None for y, t in zip(tx.year, tx.to_team)]
    mlb_team = tx.to_lvl == "MLB"
    # MLB-club events: IL placements and call-ups, keyed by MLB team id (= the org id)
    il = tx[mlb_team & (tx.code == "SC") & tx.desc.str.contains("injured list") & tx.desc.str.contains("placed")].copy()
    il["pitch"] = il.desc.str.contains("|".join(PITCH))
    il = il.assign(org=il.to_team.astype(int))
    up = tx[mlb_team & tx.code.isin(["SE", "CU"])].assign(org=lambda d: d.to_team.astype(int))
    # affiliate assignments define the current level
    asg = tx[tx.to_lvl.isin(["AA", "AAA", "A+", "A", "A-", "RK", "MLB"])][["mlbam", "date", "to_lvl", "org", "code"]]
    ids = sorted(asg[asg.to_lvl.isin(["AA", "AAA"])].mlbam.unique().tolist())
    pp = people(ids).set_index("id")
    deb = pd.to_datetime(pp.mlbDebutDate, errors="coerce")
    bd = pd.to_datetime(pp.birthDate, errors="coerce")
    is_p = pp.pos.isin(["P", "RHP", "LHP", "SP", "RP"])
    f40 = tx[tx.code.isin(ADD40 | DROP40)][["mlbam", "date", "code"]]
    opt = tx[tx.code == "OPT"][["mlbam", "date"]]

    rows = []
    for Y in SEASONS:
        snaps = pd.date_range(f"{Y}-04-07", f"{Y}-09-22", freq="W-MON")
        a = asg[asg.date < pd.Timestamp(f"{Y}-10-01")].sort_values("date")
        for t in snaps:
            cut = t - pd.Timedelta(days=1)
            cur = a[a.date <= cut].groupby("mlbam").tail(1)
            cur = cur[cur.to_lvl.isin(["AA", "AAA"]) & (cur.date.dt.year >= Y - 1)]
            m = cur.mlbam.to_numpy()
            d_ = deb.reindex(m).to_numpy()
            ok = pd.isna(d_) | (d_ > np.datetime64(t + pd.Timedelta(days=2)))
            cur, m = cur[ok], m[ok]
            if not len(cur):
                continue
            e40 = f40[(f40.date <= cut) & f40.mlbam.isin(m)].groupby("mlbam").tail(1).set_index("mlbam").code
            ilw = il[(il.date > t - pd.Timedelta(days=30)) & (il.date <= cut)]
            upw = up[(up.date > t - pd.Timedelta(days=14)) & (up.date <= cut)]
            ip = is_p.reindex(m).fillna(False).to_numpy()
            org = cur.org.to_numpy()
            def cnt(df, days, pitch=None):
                w = df[df.date > t - pd.Timedelta(days=days)]
                if pitch is None:
                    c = w.groupby("org").size()
                    return np.array([c.get(o, 0) for o in org])
                cp, cn = w[w.pitch].groupby("org").size(), w[~w.pitch].groupby("org").size()
                return np.array([(cp if q else cn).get(o, 0) for o, q in zip(org, pitch)])
            dd = deb.reindex(m).to_numpy()
            y = (~pd.isna(dd)) & (dd > np.datetime64(t + pd.Timedelta(days=2))) & (dd <= np.datetime64(t + pd.Timedelta(days=30)))
            rows.append(pd.DataFrame({
                "mlbam": m, "season": Y, "t": t, "org": org,
                "lvl_aaa": (cur.to_lvl == "AAA").to_numpy().astype(int),
                "days_at_lvl": (t - cur.date).dt.days.to_numpy(),
                "days_season": (t - pd.Timestamp(f"{Y}-04-01")).days,
                "on40": np.array([int(e40.get(x) in ADD40) if x in e40.index else 0 for x in m]),
                "ever_opt": np.isin(m, opt[opt.date <= cut].mlbam.unique()).astype(int),
                "org_il14_same": cnt(ilw, 14, ip), "org_il30_same": cnt(ilw, 30, ip), "org_il14_all": cnt(ilw, 14),
                "org_callups14": cnt(upw, 14),
                "sept": int(t.month == 9),
                "age": ((t - bd.reindex(m)).dt.days / 365.25).to_numpy(),
                "is_pitcher": ip.astype(int),
                "y": y.astype(int), "debut": dd}))
        print(f"[callup] {Y}: {sum(len(r) for r in rows if r.season.iloc[0] == Y):,} rows", flush=True)
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(PANEL, index=False)
    print(f"[callup] panel {len(panel):,} rows, {panel.mlbam.nunique():,} players, positives {int(panel.y.sum()):,}")


FEATS = ["lvl_aaa", "days_at_lvl", "days_season", "on40", "ever_opt", "org_il14_same", "org_il30_same",
         "org_il14_all", "org_callups14", "sept", "age", "is_pitcher", "p_deb1"]


def attach_pdeb(panel):
    """walk-forward season-end P(debut next season) from light_model.py, previous season end"""
    lp = RAW / "light_preds.csv"
    if not lp.exists():
        return panel
    import sqlite3
    pr = pd.read_csv(lp)[["player_id", "S", "p_deb1"]]
    con = sqlite3.connect(REPO / "prospects.db")
    xm = pd.read_sql("SELECT player_id, mlbam_id FROM prospects WHERE mlbam_id IS NOT NULL", con)
    xm["mlbam"] = pd.to_numeric(xm.mlbam_id, errors="coerce")
    pr = pr.merge(xm[["player_id", "mlbam"]], on="player_id").assign(season=lambda d: d.S + 1)
    return panel.merge(pr[["mlbam", "season", "p_deb1"]].drop_duplicates(["mlbam", "season"]),
                       on=["mlbam", "season"], how="left")


def fit():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    p = attach_pdeb(pd.read_parquet(PANEL))
    feats = [f for f in FEATS if f in p.columns]
    out = []
    for Y in range(2014, 2026):
        tr, te = p.season < Y, p.season == Y
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=100, l2_regularization=1.0, random_state=0)
        clf.fit(p.loc[tr, feats], p.loc[tr, "y"])
        d = p.loc[te, [c for c in ["mlbam", "season", "t", "y", "debut", "on40", "lvl_aaa", "p_deb1"] if c in p.columns]].copy()
        d["p_call30"] = clf.predict_proba(p.loc[te, feats])[:, 1]
        out.append(d)
        top = d.nlargest(max(1, int(len(d) * 0.01)), "p_call30")
        print(f"{Y}: n {len(d):,} base {d.y.mean():.3f} AP {average_precision_score(d.y, d.p_call30):.3f} | "
              f"top-1% precision {top.y.mean():.2f} | on40&AAA base {d[(d.on40 == 1) & (d.lvl_aaa == 1)].y.mean():.3f}", flush=True)
    pd.concat(out).to_csv(RAW / "callup_preds.csv", index=False)


if __name__ == "__main__":
    {"build": build, "fit": fit}[sys.argv[1]]()
