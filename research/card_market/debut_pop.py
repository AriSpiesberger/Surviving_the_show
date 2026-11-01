"""Player-level debut pop: which debuts pop more? (2026-10-22, user: "certain players debut with much
higher EV; team, position, all the things matter; even a lasso is nice")

Target, one row per debut with a priced first auto (2021-2026 debuts):
    pop = log( max(price in debut month, month after) / price the month before debut )
from the SportsCardsPro monthly series (raw, and PSA 10 separately). The day-level days 0-3 pop
(study_strategies.pop) exists for only ~56 cards; it is used to check that the monthly model ranks
the day-level pops too.

Features, all known before the debut: pre-debut price and its 6-month trend, PSA 10 / raw ratio,
prospect rank (best and latest overall rank, org rank, ever Top-100, Top-25), draft round / pick /
IFA / bonus, position group, bats/throws, age at debut, debut month, debut team (one-hot), card
product and age of the card, last minor-league season performance percentiles and level, the
walk-forward season-end P(debut) (a "surprise" call-up should pop more) and debut-year effects.

Model: standardised lasso (LassoCV inside each fold), leave-one-debut-year-out. Reported against a
constant and the price-band-only model the EV currently uses.

    python debut_pop.py            # fit, CV report, coefficients -> raw/debut_pop_coefs.csv
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import study_strategies as ss
from load import DB, HERE, tables
from panel import player_panel, wide

POS_GROUP = {"P": "P", "SP": "P", "RP": "P", "C": "C", "1B": "CI", "3B": "CI", "2B": "MI", "SS": "MI",
             "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "DH", "IF": "MI", "UT": "MI", "TWP": "P"}
# team market: approx. 2020 metro (CSA/CMA) population, millions; a shared market gets the whole metro.
# A mechanical definition on purpose: a hand-picked "big market" list cross-validated better (rho 0.31)
# but partly by choosing fanbase teams after seeing them; log metro gives rho 0.21 -> 0.24.
METRO = {"NYY": 23.6, "NYM": 23.6, "LAD": 18.7, "LAA": 18.7, "CHC": 9.9, "CWS": 9.9, "WSH": 10.0, "SF": 9.6,
         "OAK": 9.6, "ATH": 9.6, "BOS": 8.3, "TEX": 8.2, "HOU": 7.3, "PHI": 7.4, "MIA": 6.9, "ATL": 6.9, "TOR": 6.2,
         "DET": 5.4, "SEA": 5.0, "MIN": 4.1, "AZ": 5.0, "SD": 3.3, "COL": 3.6, "STL": 2.9, "TB": 3.2, "BAL": 2.9,
         "PIT": 2.7, "CLE": 3.6, "CIN": 2.3, "KC": 2.5, "MIL": 2.0}
PERF = ["pct_woba", "pct_iso", "pct_k_pct", "pct_bb_pct", "pct_era", "pct_k9", "pct_bb9", "pct_fip", "pct_whip"]


@lru_cache
def debuts(grade: str = "raw") -> pd.DataFrame:
    px = np.exp(wide(player_panel(grade=grade))).sort_index(axis=1).ffill(axis=1, limit=2)
    ppl = tables()["people"].set_index("player_id")
    d = ppl[ppl.debut_date.notna() & (ppl.debut_date >= "2021-04-01")].copy()
    d = d[d.index.isin(px.index)]
    dm = d.debut_date.dt.to_period("M")
    rows = []
    for pid, m in dm.items():
        pre, at, nxt, six = m - 1, m, m + 1, m - 7
        if pre not in px.columns or at not in px.columns:
            continue
        p_pre = px.at[pid, pre]
        post = np.nanmax([px.at[pid, at], px.at[pid, nxt] if nxt in px.columns else np.nan])
        if not (p_pre > 0 and post > 0):
            continue
        rows.append({"player_id": pid, "debut_month": m, "p_pre": p_pre, "pop": np.log(post / p_pre),
                     "trend6": np.log(p_pre / px.at[pid, six]) if six in px.columns and px.at[pid, six] > 0 else np.nan})
    return pd.DataFrame(rows)


def features(d: pd.DataFrame) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    ppl = tables()["people"].set_index("player_id")
    pr = pd.read_sql("select player_id, primary_position, bats, throws, draft_round, draft_pick, signing_bonus_usd, "
                     "is_international, birth_date from prospects", con).set_index("player_id")
    x = d.join(pr, on="player_id").join(ppl[["is_pitcher"]], on="player_id")
    dy = x.debut_month.dt.year
    x["debut_year"] = dy
    x["month"] = x.debut_month.dt.month
    x["sept"] = (x.month >= 9).astype(float)
    x["april"] = (x.month <= 4).astype(float)
    x["age"] = (x.debut_month.dt.to_timestamp() - pd.to_datetime(x.birth_date)).dt.days / 365.25
    x["log_p_pre"] = np.log(x.p_pre)
    x["round"] = x.draft_round.fillna(25).clip(upper=25)
    x["r1"] = (x.draft_round == 1).astype(float)
    x["log_pick"] = np.log1p(x.draft_pick.fillna(600))
    x["ifa"] = x.is_international.fillna(0).astype(float)
    x["log_bonus"] = np.log1p(x.signing_bonus_usd.fillna(0))
    x["pos"] = x.primary_position.map(POS_GROUP).fillna(pd.Series(np.where(x.is_pitcher == 1, "P", "OF"), index=x.index))
    x["bats_L"] = (x.bats == "L").astype(float)
    x["bats_S"] = (x.bats == "S").astype(float)
    x["throws_L"] = (x.throws == "L").astype(float)
    # prospect rank before debut
    rh = pd.read_sql("select player_id, as_of, overall_rank, org_rank from rankings_history", con)
    rh["as_of"] = pd.to_datetime(rh.as_of, errors="coerce")
    rh = rh.merge(x[["player_id", "debut_month"]], on="player_id")
    rh = rh[rh.as_of < rh.debut_month.dt.to_timestamp()]
    g = rh.sort_values("as_of").groupby("player_id")
    x = x.join(pd.DataFrame({"best_rank": g.overall_rank.min(), "last_rank": g.overall_rank.last(),
                             "best_org_rank": g.org_rank.min(), "last_org_rank": g.org_rank.last()}), on="player_id")
    pk = pd.read_sql("select player_id, year, rank from prospect_rankings", con)
    pk = pk.merge(x[["player_id", "debut_year"]], on="player_id")
    pk = pk[pk.year <= pk.debut_year].groupby("player_id")["rank"].min().rename("top_list_best")
    x = x.join(pk, on="player_id")
    x["ever_top100"] = (x.top_list_best.notna() | (x.best_rank <= 100)).astype(float)
    x["top25"] = ((x.top_list_best <= 25) | (x.best_rank <= 25)).astype(float)
    best = x[["best_rank", "top_list_best"]].min(axis=1)
    x["log_best_rank"] = np.log(best.fillna(400).clip(upper=400))
    x["log_last_rank"] = np.log(x.last_rank.fillna(400).clip(upper=400))
    x["log_org_rank"] = np.log(x.last_org_rank.fillna(40).clip(upper=40))
    # debut team = org on his first MLB season row; else current org
    mlb = pd.read_sql("select player_id, season_year, org from season_stats where level='MLB' and org is not null", con)
    mlb = mlb.sort_values("season_year").drop_duplicates("player_id").set_index("player_id").org
    cur = pd.read_sql("select player_id, current_org from prospects", con).set_index("player_id").current_org
    x["team"] = x.player_id.map(mlb).fillna(x.player_id.map(cur)).fillna("UNK")
    x["log_metro"] = np.log(x.team.map(_team_code).map(METRO))
    # card: product and age
    c = tables()["cards"]
    c = c[(c.kind == "first_auto") & c.player_id.notna()].drop_duplicates("player_id").set_index("player_id")
    x["draft_product"] = x.player_id.map(c.set.str.contains("Draft", case=False)).fillna(False).astype(float)
    x["card_age"] = x.debut_year - x.player_id.map(c.card_year)
    # PSA 10 / raw ratio before debut
    p10 = np.exp(wide(player_panel(grade="psa10"))).sort_index(axis=1).ffill(axis=1, limit=2)
    raw = np.exp(wide(player_panel(grade="raw"))).sort_index(axis=1).ffill(axis=1, limit=2)
    def _at(tab, pid, m):
        return tab.at[pid, m] if pid in tab.index and m in tab.columns else np.nan
    x["log_psa_ratio"] = [np.log(_at(p10, p, m - 1) / _at(raw, p, m - 1)) for p, m in zip(x.player_id, x.debut_month)]
    # last minor-league season before debut (highest level that year)
    ms = pd.read_sql(f"select player_id, season_year, level, pa, ip, {', '.join(PERF)} from season_stats "
                     "where level in ('AAA','AA','A+','A')", con)
    ms = ms.merge(x[["player_id", "debut_year"]], on="player_id")
    ms = ms[ms.season_year < ms.debut_year]
    ms["lv"] = ms.level.map({"AAA": 3, "AA": 2, "A+": 1, "A": 0})
    ms = ms.sort_values(["season_year", "lv"]).groupby("player_id").tail(1).set_index("player_id")
    x["last_level"] = x.player_id.map(ms.lv)
    x["skipped_aaa"] = (x.last_level < 3).astype(float)
    for col in PERF:
        x[col] = x.player_id.map(ms[col])
    hit = x.pos != "P"
    x["perf_bat"] = np.where(hit, x[["pct_woba", "pct_iso"]].mean(axis=1) - x.pct_k_pct.fillna(50) * 0, np.nan)
    x["perf_pit"] = np.where(~hit, x[["pct_k9", "pct_fip"]].mean(axis=1), np.nan)
    # walk-forward "surprise": P(debut next season) at the prior season end
    lp = pd.read_csv(HERE / "raw" / "light_preds.csv")[["player_id", "S", "p_deb1", "p_top100"]]
    lp = lp.rename(columns={"S": "debut_year_minus1"})
    x["debut_year_minus1"] = x.debut_year - 1
    x = x.merge(lp, on=["player_id", "debut_year_minus1"], how="left")
    x["p_deb1"] = x.p_deb1
    return x


def _team_code(t):
    """season_stats.org is an abbreviation; prospects.current_org is a full name"""
    if t in METRO:
        return t
    names = {"Yankees": "NYY", "Mets": "NYM", "Dodgers": "LAD", "Angels": "LAA", "Cubs": "CHC", "White Sox": "CWS",
             "Nationals": "WSH", "Giants": "SF", "Athletics": "ATH", "Red Sox": "BOS", "Rangers": "TEX", "Astros": "HOU",
             "Phillies": "PHI", "Marlins": "MIA", "Braves": "ATL", "Blue Jays": "TOR", "Tigers": "DET", "Mariners": "SEA",
             "Twins": "MIN", "Diamondbacks": "AZ", "Padres": "SD", "Rockies": "COL", "Cardinals": "STL", "Rays": "TB",
             "Orioles": "BAL", "Pirates": "PIT", "Guardians": "CLE", "Reds": "CIN", "Royals": "KC", "Brewers": "MIL"}
    for k, v in names.items():
        if isinstance(t, str) and t.endswith(k):
            return v
    return None


NUM = ["log_metro", "log_p_pre", "trend6", "log_psa_ratio", "sept", "april", "age", "round", "r1", "log_pick", "ifa", "log_bonus",
       "bats_L", "bats_S", "throws_L", "ever_top100", "top25", "log_best_rank", "log_last_rank", "log_org_rank",
       "draft_product", "card_age", "last_level", "skipped_aaa", "perf_bat", "perf_pit", "p_deb1", "p_top100"] + PERF
CAT = ["pos", "team", "debut_year"]


def design(x: pd.DataFrame, cols=None, med=None) -> pd.DataFrame:
    X = x[NUM].copy()
    for c in NUM:  # missing -> median (the training median when given) + indicator
        X[f"{c}_na"] = X[c].isna().astype(float)
        X[c] = X[c].fillna(X[c].median() if med is None else med.get(c, 0.0))
    X = pd.concat([X, pd.get_dummies(x[CAT].astype(str), prefix=CAT, dtype=float)], axis=1)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


def cv(grade: str = "raw", min_p: float = 3.0):
    d = debuts(grade)
    d = d[d.p_pre >= min_p]
    x = features(d).dropna(subset=["pop"])
    x = x[x.debut_year <= 2026]
    X = design(x)
    y = x["pop"].clip(-1.5, 2.5).to_numpy()
    pred = np.full(len(x), np.nan)
    band = pd.cut(x.p_pre, ss.BANDS, labels=ss.BAND_LBL, right=False).astype(str)
    pred_band = np.full(len(x), np.nan)
    for yr in sorted(x.debut_year.unique()):
        te = (x.debut_year == yr).to_numpy()
        if te.sum() < 10:
            continue
        Xtr = X[~te].drop(columns=[c for c in X if c.startswith("debut_year_")])
        Xte = X[te].drop(columns=[c for c in X if c.startswith("debut_year_")])
        # year effects: demean y by training-year mean so the held-out year gets the average level
        ytr = y[~te]
        m = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0)).fit(Xtr, ytr)
        pred[te] = m.predict(Xte)
        bm = pd.Series(ytr).groupby(band[~te].to_numpy()).mean()
        pred_band[te] = band[te].map(bm).fillna(ytr.mean()).to_numpy()
    ok = ~np.isnan(pred)
    x = x.assign(pred=pred, pred_band=pred_band)[ok]
    yy = y[ok]
    def r2(p):
        return 1 - ((yy - p) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    print(f"\n===== {grade}: {ok.sum()} debuts, leave-one-debut-year-out =====")
    print(f"R2 lasso {r2(x.pred.to_numpy()):.3f}   R2 price-band {r2(x.pred_band.to_numpy()):.3f}")
    print("spearman lasso", round(spearmanr(x.pred, yy)[0], 3), " band", round(spearmanr(x.pred_band, yy)[0], 3))
    x["y"] = yy
    x["q"] = pd.qcut(x.pred, 5, labels=False)
    print(x.groupby("q").agg(n=("y", "size"), pred=("pred", "mean"), real=("y", "mean"), real_mult=("y", lambda v: np.exp(v).mean()),
                             med_price=("p_pre", "median")).round(3).to_string())
    print("by year:", {yr: round(spearmanr(g.pred, g.y)[0], 2) for yr, g in x.groupby("debut_year") if len(g) > 10})
    # full fit for coefficients
    Xf = X.drop(columns=[c for c in X if c.startswith("debut_year_")])
    mf = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0)).fit(Xf, y)
    coef = pd.Series(mf[-1].coef_, index=Xf.columns)
    coef = coef[coef != 0].sort_values(key=abs, ascending=False)
    print("nonzero coefficients (per 1 sd):\n", coef.round(3).head(30).to_string())
    coef.to_csv(HERE / "raw" / f"debut_pop_coefs_{grade}.csv")
    return x, mf, Xf.columns


# the variant that cross-validated best (leave-one-debut-year-out R2 0.037, rho 0.21, top vs bottom
# quintile +0.47 vs +0.20 log): team one-hots and the long performance list only add noise at n~350
CORE = ["log_metro", "log_p_pre", "top25", "ever_top100", "log_best_rank", "sept", "april", "card_age", "log_psa_ratio", "age",
        "r1", "ifa", "skipped_aaa", "p_deb1"]


def core_cols(X):
    return CORE + [c for c in X if c.startswith("pos_")]


@lru_cache
def train_set(grade: str = "raw"):
    d = debuts(grade)
    d = d[d.p_pre >= 3]
    x = features(d).reset_index(drop=True)
    return x, x["pop"].clip(-1.5, 2.5).to_numpy()


def fit_core(grade: str = "raw", exclude_year: int | None = None):
    """Lasso on CORE; returns a predict(frame) closure. exclude_year leaves a debut year out."""
    x, y = train_set(grade)
    keep = (x.debut_year != exclude_year).to_numpy() if exclude_year else np.ones(len(x), bool)
    med = x.loc[keep, NUM].median()
    X = design(x[keep], med=med)
    cols = core_cols(X)
    m = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0)).fit(X[cols], y[keep])
    timing = x.loc[keep, ["sept", "april"]].mean()

    def predict(f: pd.DataFrame, neutral_timing: bool = True) -> np.ndarray:
        f = f.copy()
        if neutral_timing:  # debut month is unknown at the buy: use the training mix
            f["sept"], f["april"] = timing["sept"], timing["april"]
        Xf = design(f, med=med)
        for c in cols:
            if c not in Xf:
                Xf[c] = 0.0
        return m.predict(Xf[cols])
    return predict


def check_daylevel(x: pd.DataFrame):
    """does the monthly-pop model rank the day-level days 0-3 pops?"""
    t = tables()
    c = t["cards"]
    c = c[(c.kind == "first_auto") & c.player_id.notna()]
    s = t["sales"]
    s = s[s.grade == "raw"].merge(c[["id", "player_id"]], on="id").merge(t["people"][["player_id", "debut_date"]], on="player_id")
    s = s[s.debut_date.notna()]
    s["day"] = (s.date - s.debut_date).dt.days
    base = s[s.day.between(-90, -15)].groupby("player_id").price.median()
    w = s[s.day.between(0, 3)].groupby("player_id").price.median()
    dl = np.log(w / base).dropna().rename("day_pop")
    j = x.set_index("player_id").join(dl, how="inner")
    if len(j) > 5:
        print(f"\nday-level check: n={len(j)}, spearman(pred monthly, day 0-3 pop) = {spearmanr(j.pred, j.day_pop)[0]:.3f}, "
              f"monthly vs day = {spearmanr(j.y, j.day_pop)[0]:.3f}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for g in ("raw", "psa10"):
        x, _, _ = cv(g)
        if g == "raw":
            check_daylevel(x)


# ---------------------------------------------------------------------------
# Robustness round (2026-10-22): a less noisy target and more pre-debut information.
# ---------------------------------------------------------------------------
GRADES = ("raw", "psa9", "psa10")


@lru_cache
def composite() -> pd.DataFrame:
    """One pop per player from every grade he has: each grade's pop centred on that grade's mean,
    averaged, then put back on the raw scale. Grades agree at rho 0.44-0.57, so averaging removes
    card-level noise while keeping the player-level component."""
    parts = []
    for g in GRADES:
        d = debuts(g)
        d = d[d.p_pre >= 3].set_index("player_id")
        parts.append((d["pop"].clip(-1.5, 2.5) - d["pop"].clip(-1.5, 2.5).mean()).rename(g))
    j = pd.concat(parts, axis=1)
    raw = debuts("raw").set_index("player_id")
    raw = raw[raw.p_pre >= 3]
    out = pd.DataFrame({"pop_comp": j.mean(axis=1) + raw["pop"].clip(-1.5, 2.5).mean(), "n_grades": j.notna().sum(axis=1)})
    return out


def extra_features(x: pd.DataFrame) -> pd.DataFrame:
    """scouting summary as of the season before debut, price liquidity and card supply"""
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    from prospects import config
    sc = pd.read_csv(config.SCOUTING_GRADES, low_memory=False, usecols=lambda c: c in {"player_id", "season", "source", "fv", "ovr_rank", "eta", "risk"})
    sc = sc.sort_values(["player_id", "season", "source"]).drop_duplicates(["player_id", "season"])
    x = x.copy()
    x["_k"] = x.debut_year - 1
    sc = sc.rename(columns={"season": "_k"})
    x = pd.merge_asof(x.sort_values("_k"), sc.sort_values("_k"), on="_k", by="player_id", direction="backward")
    x["scout_fv"] = x.fv
    x["scout_eta_gap"] = x.eta - x.debut_year          # > 0: arrived ahead of the scouts' ETA
    x["scout_risk"] = x.risk
    x["log_scout_rank"] = np.log(x.ovr_rank.clip(lower=1, upper=400))
    # liquidity: share of the 12 months before debut in which the raw price moved
    m = player_panel(grade="raw")
    mv = m.assign(moved=m.ret.ne(0) & m.ret.notna()).set_index(["player_id", "month"]).moved
    x["liq12"] = [np.mean([mv.get((p, mo - k), False) for k in range(1, 13)]) for p, mo in zip(x.player_id, x.debut_month)]
    # supply: how many first-auto cards (products) exist for him
    c = tables()["cards"]
    x["n_first_autos"] = x.player_id.map(c[c.kind == "first_auto"].groupby("player_id").size()).fillna(0)
    return x.drop(columns=["_k", "fv", "eta", "risk", "ovr_rank"])


EXTRA = ["scout_fv", "scout_eta_gap", "scout_risk", "log_scout_rank", "liq12", "n_first_autos"]


def experiment():
    from sklearn.linear_model import LassoCV
    x, _ = train_set("raw")
    x = extra_features(x).merge(composite(), left_on="player_id", right_index=True, how="left")
    for c in EXTRA:
        if c not in NUM:
            NUM.append(c)
    med = x[NUM].median()
    X = design(x, med=med)
    base = core_cols(X)
    targets = {"raw pop": x["pop"].clip(-1.5, 2.5).to_numpy(), "composite": x.pop_comp.to_numpy()}
    for tname, y in targets.items():
        for fname, cols in (("core", base), ("core+extra", base + EXTRA)):
            pred = np.full(len(x), np.nan)
            for yr in sorted(x.debut_year.unique()):
                te = (x.debut_year == yr).to_numpy() & ~np.isnan(y)
                tr = (x.debut_year != yr).to_numpy() & ~np.isnan(y)
                if te.sum() < 10:
                    continue
                mdl = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0)).fit(X.loc[tr, cols], y[tr])
                pred[te] = mdl.predict(X.loc[te, cols])
            ok = ~np.isnan(pred)
            # always score against BOTH targets: a model trained on the composite must still rank raw pops
            for sname, s in targets.items():
                k = ok & ~np.isnan(s)
                print(f"train {tname:9s} feats {fname:10s} -> rho vs {sname:9s} {spearmanr(pred[k], s[k])[0]:+.3f}  (n={k.sum()})")
    yy = targets["composite"]
    k = ~np.isnan(yy)
    mdl = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0)).fit(X.loc[k, base + EXTRA], yy[k])
    print(pd.Series(mdl[-1].coef_, base + EXTRA).round(3).sort_values().to_string())


@lru_cache
def debuts_smooth(grade: str = "raw") -> pd.DataFrame:
    """pop against the median of the three months before debut, and the price six months out as a
    feature, so the feature and the target do not share the month -1 price's noise"""
    px = np.exp(wide(player_panel(grade=grade))).sort_index(axis=1).ffill(axis=1, limit=2)
    ppl = tables()["people"]
    d = ppl[ppl.debut_date.notna() & (ppl.debut_date >= "2021-04-01") & ppl.player_id.isin(px.index)]
    rows = []
    for pid, dd in zip(d.player_id, d.debut_date):
        m = dd.to_period("M")
        pre = [px.at[pid, m - k] for k in (1, 2, 3) if (m - k) in px.columns]
        post = [px.at[pid, k] for k in (m, m + 1) if k in px.columns]
        six = px.at[pid, m - 6] if (m - 6) in px.columns else np.nan
        if len(pre) < 2 or not post or not np.nanmax(post) > 0:
            continue
        base = np.nanmedian(pre)
        if not base > 0:
            continue
        rows.append({"player_id": pid, "debut_month": m, "p_pre": base, "p6": six,
                     "pop": np.log(np.nanmax(post) / base)})
    return pd.DataFrame(rows)


def experiment_smooth():
    from sklearn.linear_model import LassoCV
    res = {}
    for g in GRADES:
        d = debuts_smooth(g)
        res[g] = d[d.p_pre >= 3].set_index("player_id")["pop"].clip(-1.5, 2.5)
    j = pd.DataFrame(res)
    print("smoothed-baseline pops, cross-grade spearman:\n", j.corr(method="spearman").round(2))
    comp = (j - j.mean()).mean(axis=1) + j["raw"].mean()
    d = debuts_smooth("raw")
    d = d[d.p_pre >= 3]
    x = extra_features(features(d.drop(columns=["p6"]).assign(trend6=np.nan)).merge(d[["player_id", "p6"]], on="player_id"))
    x["log_p6"] = np.log(x.p6)
    for c in EXTRA + ["log_p6"]:
        if c not in NUM:
            NUM.append(c)
    X = design(x, med=x[NUM].median())
    nopx = [c for c in core_cols(X) if c not in ("log_p_pre", "log_psa_ratio")]
    sets = {"core (month -1 price)": core_cols(X) + EXTRA, "price from month -6": nopx + EXTRA + ["log_p6"],
            "no price at all": nopx + EXTRA}
    targets = {"raw": x["pop"].clip(-1.5, 2.5).to_numpy(), "composite": x.player_id.map(comp).to_numpy()}
    for tn, y in targets.items():
        for sn, cols in sets.items():
            pred = np.full(len(x), np.nan)
            for yr in sorted(x.debut_year.unique()):
                te = (x.debut_year == yr).to_numpy() & ~np.isnan(y)
                tr = (x.debut_year != yr).to_numpy() & ~np.isnan(y)
                if te.sum() < 10:
                    continue
                mdl = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0)).fit(X.loc[tr, cols], y[tr])
                pred[te] = mdl.predict(X.loc[te, cols])
            k = ~np.isnan(pred)
            print(f"target {tn:9s} | {sn:22s} rho {spearmanr(pred[k], y[k])[0]:+.3f}  n={k.sum()}")
    y = targets["composite"]; k = ~np.isnan(y)
    mdl = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0)).fit(X.loc[k, sets["price from month -6"]], y[k])
    print(pd.Series(mdl[-1].coef_, sets["price from month -6"]).round(3).sort_values().to_string())
    return x, comp


# ---------------------------------------------------------------------------
# Day-level pops from parallels (2026-10-24 scrape, in-browser, raw/variant_daypop.csv)
# ---------------------------------------------------------------------------
@lru_cache
def day_pops() -> pd.DataFrame:
    """Per player: the days 0-3 pop measured on every card that has sales on both sides of the debut.
    Base first autos come from the sales already scraped (load.tables); parallels (refractors,
    numbered colours) from the 6,057-page parallel scrape, each card against its own median sale
    15-90 days before. Combined as an n-weighted mean of the per-source medians."""
    import json
    v = pd.read_csv(HERE / "raw" / "variant_daypop.csv")
    work = json.load(open(HERE.parents[1] / "runs" / "experiments" / "ev_20261022" / "variant_work_f.json"))
    v["player_id"] = v.i.map(lambda i: work[i][0])
    par = v.groupby("player_id").apply(lambda g: pd.Series({
        "par_n": g.n_d.sum(), "par_pop": np.average(g.med_d, weights=g.n_d),
        "par_n_late": g.n_l.sum(), "par_late": np.average(g.med_l.fillna(0), weights=g.n_l) if g.n_l.sum() else np.nan,
        "par_n_pre": g.n_p.sum(), "par_pre": np.average(g.med_p.fillna(0), weights=g.n_p) if g.n_p.sum() else np.nan}),
        include_groups=False)
    t = tables()
    c = t["cards"]
    c = c[(c.kind == "first_auto") & c.player_id.notna() & ~c.code.fillna("").str.startswith(("CPS", "BMA"))]
    s = t["sales"]
    s = s[s.grade == "raw"].merge(c[["id", "player_id"]], on="id").merge(t["people"][["player_id", "debut_date"]], on="player_id")
    s = s[s.debut_date.notna()]
    s["day"] = (s.date - s.debut_date).dt.days
    base = s[s.day.between(-90, -15)].groupby("id").price.median().rename("b")
    w = s[s.day.between(0, 3)].groupby(["id", "player_id"]).price.median().rename("w").reset_index().join(base, on="id").dropna()
    w["pop"] = np.log(w.w / w.b)
    bp = w.groupby("player_id")["pop"].agg(base_pop="median", base_n="size")
    d = par.join(bp, how="outer")
    num = d.par_pop.fillna(0) * d.par_n.fillna(0) + d.base_pop.fillna(0) * d.base_n.fillna(0)
    den = d.par_n.fillna(0) + d.base_n.fillna(0)
    d["day_pop"] = (num / den).clip(-1.5, 3.0)
    d["day_n"] = den
    return d[den > 0]


def experiment_day(min_n: int = 1):
    """Lasso on the day-level pop (base + parallels), leave one debut year out. Features avoid the
    month -1 price (shared noise); the card's price six months out stands in for price level."""
    d = day_pops()
    d = d[d.day_n >= min_n]
    sm = debuts_smooth("raw").set_index("player_id")
    ppl = tables()["people"].set_index("player_id")
    base = pd.DataFrame({"player_id": d.index, "debut_month": ppl.debut_date.reindex(d.index).dt.to_period("M").to_numpy(),
                         "p_pre": sm.p_pre.reindex(d.index).to_numpy(), "pop": d.day_pop.to_numpy(), "trend6": np.nan})
    base = base.dropna(subset=["debut_month"]).reset_index(drop=True)
    x = extra_features(features(base))
    x["log_p6"] = np.log(x.player_id.map(sm.p6))
    x["w"] = x.player_id.map(d.day_n).clip(upper=6)
    for c in EXTRA + ["log_p6"]:
        if c not in NUM:
            NUM.append(c)
    X = design(x, med=x[NUM].median())
    nopx = [c for c in core_cols(X) if c not in ("log_p_pre", "log_psa_ratio")]
    sets = {"decoupled (p6)": nopx + EXTRA + ["log_p6"], "no price": nopx + EXTRA,
            "few: metro,eta,sept,liq,p6,age,top25": ["log_metro", "scout_eta_gap", "sept", "liq12", "log_p6", "age", "top25"]}
    y = x["pop"].to_numpy()
    print(f"n={len(x)} debuts with a day-level pop; years {x.debut_year.value_counts().sort_index().to_dict()}")
    for sn, cols in sets.items():
        pred = np.full(len(x), np.nan)
        for yr in sorted(x.debut_year.unique()):
            te = (x.debut_year == yr).to_numpy()
            if te.sum() < 8:
                continue
            mdl = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0))
            mdl.fit(X.loc[~te, cols], y[~te], lassocv__sample_weight=x.w[~te].to_numpy())
            pred[te] = mdl.predict(X.loc[te, cols])
        k = ~np.isnan(pred)
        q = pd.qcut(pred[k], 3, labels=False)
        print(f"{sn:40s} rho {spearmanr(pred[k], y[k])[0]:+.3f}  n={k.sum()}  tercile mean pop: "
              + " / ".join(f"{np.exp(y[k][q == t]).mean():.2f}x" for t in range(3)))
    cols = sets["decoupled (p6)"]
    mdl = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0)).fit(X[cols], y, lassocv__sample_weight=x.w.to_numpy())
    print(pd.Series(mdl[-1].coef_, cols).round(3).loc[lambda s: s != 0].sort_values().to_string())
    # univariate checks on the bigger sample
    for col in ["sept", "top25", "ever_top100", "ifa", "skipped_aaa"]:
        g = x.groupby(col)["pop"].agg(["size", "mean"])
        print(col, {k: (int(r["size"]), round(float(np.exp(r["mean"])), 2)) for k, r in g.iterrows()})
    print("metro tercile:", x.groupby(pd.qcut(x.log_metro, 3, labels=["small", "mid", "big"]), observed=True)["pop"].agg(lambda v: round(float(np.exp(v).mean()), 2)).to_dict())
    print("eta gap:", x.groupby(pd.cut(x.scout_eta_gap, [-9, -0.5, 0.5, 9], labels=["late", "on time", "early"]), observed=True)["pop"].agg(["size", lambda v: round(float(np.exp(v).mean()), 2)]).to_dict())
    return x
