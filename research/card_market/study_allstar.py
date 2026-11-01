"""First-All-Star breakout trade (2026-10-25).

Event study (study_ceiling.py): a first All-Star season lifts the raw first auto +88% vs the market
over the calendar year (+39% Jan->Jun, +17% around the July selection). The prospect model's
P(star) is a 3-6 year number and loses as a buy signal. This asks the near-term question instead:

    each January of season Y, among young MLB players (debuted Y-5..Y-1, never an All-Star, priced
    first auto), who makes his first All-Star team in Y?

Features, all known in January Y: last MLB season (PA, wOBA, ISO, K%, BB%; IP, ERA, FIP, K/9, BB/9,
WHIP), MLB career PA / IP, seasons since debut, age, prospect pedigree (best overall rank, ever
Top-100 / Top-25, draft round), pitcher, and the card's own price and its last-6-month move (what the
market already expects). Logistic regression (L2, standardised) and a small GBM, leave one season out.

Trade: buy the first auto in January Y (monthly SportsCardsPro price), sell in August Y (after the
selection), net of study_basket.COSTS; every pick counts, hits and misses. Labels: Stats API rosters
(raw/allstars.csv, fetch_allstars.py).

    python study_allstar.py
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import study_basket as sb
from load import DB, HERE, tables
from panel import player_panel, wide
from study_debut import boot

pd.set_option("display.width", 220)
SEASONS = (2022, 2023, 2024, 2025, 2026)
BAT = ["pa", "woba", "iso", "k_pct", "bb_pct"]
PIT = ["ip", "era", "fip", "k9", "bb9", "whip"]


def prices(grade: str = "raw") -> pd.DataFrame:
    return np.exp(wide(player_panel(grade=grade))).sort_index(axis=1).ffill(axis=1, limit=3)


def panel(grade: str = "raw") -> pd.DataFrame:
    con = sqlite3.connect(DB)
    ppl = tables()["people"].set_index("player_id")
    pr = pd.read_sql("select player_id, mlbam_id, birth_date, draft_round, is_pitcher from prospects", con).set_index("player_id")
    pr["mlbam"] = pd.to_numeric(pr.mlbam_id, errors="coerce")
    ast = pd.read_csv(HERE / "raw" / "allstars.csv")
    first_as = ast.groupby("mlbam").season.min()
    mlb = pd.read_sql(f"select player_id, season_year y, {', '.join(sorted(set(BAT + PIT)))} from season_stats where level='MLB'", con)
    mlb = mlb.groupby(["player_id", "y"]).agg({**{c: "sum" for c in ("pa", "ip")}, **{c: "mean" for c in set(BAT + PIT) - {"pa", "ip"}}}).reset_index()
    rk = pd.read_sql("select player_id, as_of, overall_rank from rankings_history", con)
    rk["as_of"] = pd.to_datetime(rk.as_of, errors="coerce")
    top = pd.read_sql("select player_id, year, rank from prospect_rankings", con)
    P = prices(grade)
    rows = []
    for Y in SEASONS:
        buy, sell = pd.Period(f"{Y}-01", "M"), pd.Period(f"{Y}-08", "M")
        if buy not in P.columns:
            continue
        d = ppl[ppl.debut_date.notna()].copy()
        dy = d.debut_date.dt.year
        d = d[(dy >= Y - 5) & (dy <= Y - 1)]
        d = d.join(pr[["mlbam", "birth_date", "draft_round", "is_pitcher"]], rsuffix="_p")
        d = d[~(d.mlbam.map(first_as) < Y)]                     # never an All-Star before Y
        d = d.join(pd.DataFrame({"p0": P[buy], "p1": P.get(sell), "p_6": P.get(buy - 6)}), how="inner").dropna(subset=["p0"])
        d["Y"] = Y
        d["y_as"] = (d.mlbam.map(first_as) == Y).astype(int)
        last = mlb[mlb.y == Y - 1].set_index("player_id")
        for c in BAT + PIT:
            d[f"last_{c}"] = d.index.map(last[c])
        car = mlb[mlb.y < Y].groupby("player_id")[["pa", "ip"]].sum()
        d["car_pa"], d["car_ip"] = d.index.map(car.pa).fillna(0), d.index.map(car.ip).fillna(0)
        d["yrs_since_debut"] = Y - dy.reindex(d.index)
        d["age"] = (pd.Timestamp(f"{Y}-07-01") - pd.to_datetime(d.birth_date)).dt.days / 365.25
        r = rk[rk.as_of < pd.Timestamp(f"{Y}-01-01")].groupby("player_id").overall_rank.min()
        t = top[top.year <= Y].groupby("player_id")["rank"].min()
        best = pd.concat([d.index.map(r).to_series(index=d.index), d.index.map(t).to_series(index=d.index)], axis=1).min(axis=1)
        d["log_best_rank"] = np.log(best.fillna(400).clip(1, 400))
        d["ever_top100"] = (best <= 100).astype(float)
        d["top25"] = (best <= 25).astype(float)
        d["round"] = d.draft_round.fillna(25).clip(upper=25)
        d["pitcher"] = d.is_pitcher.fillna(0).astype(float)
        d["log_p0"] = np.log(d.p0)
        d["trend6"] = np.log(d.p0 / d.p_6)
        rows.append(d)
    return pd.concat(rows).rename_axis("player_id").reset_index()


FEATS = [f"last_{c}" for c in BAT + PIT] + ["car_pa", "car_ip", "yrs_since_debut", "age", "log_best_rank", "ever_top100",
                                            "top25", "round", "pitcher", "log_p0", "trend6"]


def design(d, med=None):
    X = d[FEATS].copy()
    for c in FEATS:
        X[f"{c}_na"] = X[c].isna().astype(float)
        X[c] = X[c].fillna(X[c].median() if med is None else med[c])
    return X


def run(grade: str = "raw"):
    d = panel(grade)
    print(f"===== {grade}: {len(d)} player-seasons, {d.y_as.sum()} first All-Stars; by season "
          f"{d.groupby('Y').agg(n=('y_as', 'size'), hits=('y_as', 'sum')).T.to_dict()} =====")
    out = []
    for Y in sorted(d.Y.unique()):
        tr, te = d[d.Y != Y], d[d.Y == Y].copy()
        med = tr[FEATS].median()
        Xtr, Xte = design(tr, med), design(te, med)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced")).fit(Xtr, tr.y_as)
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                            class_weight="balanced", random_state=0).fit(Xtr, tr.y_as)
        te["s_lr"], te["s_gb"] = lr.predict_proba(Xte)[:, 1], gb.predict_proba(Xte)[:, 1]
        te["s_mix"] = te.s_lr.rank(pct=True) + te.s_gb.rank(pct=True)
        out.append(te)
    o = pd.concat(out)
    for s in ("s_lr", "s_gb", "s_mix", "log_p0"):
        aps = {Y: round(average_precision_score(g.y_as, g[s]), 3) for Y, g in o.groupby("Y") if g.y_as.sum()}
        print(f"{s:7s} AP by season {aps}  AUC {roc_auc_score(o.y_as, o[s]):.3f}   (base rate {o.y_as.mean():.3f})")
    o = o.dropna(subset=["p1"])
    o["net"] = sb.net(o.p0, o.p1)

    def summ(g):
        if len(g) < 3:
            return {"n": len(g)}
        lo, hi = boot(g.net) if len(g) >= 5 else (np.nan, np.nan)
        return {"n": len(g), "hit": g.y_as.mean(), "med_p0": g.p0.median(), "net": g.net.mean(), "lo": lo, "hi": hi,
                "median": g.net.median(), **{f"y{Y % 100}": g[g.Y == Y].net.mean() for Y in SEASONS}}
    tab = {"all young MLB (Jan->Aug)": o, "hits (hindsight)": o[o.y_as == 1], "misses (hindsight)": o[o.y_as == 0]}
    for k in (5, 10, 20):
        tab[f"top {k}/season by model"] = o.sort_values("s_mix", ascending=False).groupby("Y").head(k)
    for k in (10,):
        tab[f"top {k}/season by model, price >= $10"] = o[o.p0 >= 10].sort_values("s_mix", ascending=False).groupby("Y").head(k)
    print(pd.DataFrame({k: summ(g) for k, g in tab.items()}).T.round(3).to_string(), "\n")
    full = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced")).fit(design(d), d.y_as)
    print(pd.Series(full[-1].coef_[0], design(d).columns).round(2).sort_values().loc[lambda s: s.abs() > 0.1].to_string())
    return o


if __name__ == "__main__":
    for g in ("raw", "psa10"):
        run(g)
