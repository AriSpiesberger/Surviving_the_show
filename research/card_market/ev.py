"""Expected value of buying a player's first Bowman Chrome auto now (2026-10-22).

One-season trade, the §3c/§8 engine: buy in November of season Y, sell in days 0-3 after an MLB
debut in Y+1, otherwise sell at the October Y+1 monthly price. The payoff per dollar of the
reference (SportsCardsPro monthly) price is a two-path mixture

    E[exit / p_ref] = P(debut in Y+1) * M_debut(cell) + (1 - P) * M_miss(cell)

M_debut and M_miss are mean exit multiples measured on the 2022-25 November positions
(study_strategies.positions: walk-forward probabilities from light_model.py), per price band and
season-end level, shrunk toward the pooled multiple. Expected net of costs (study_basket.COSTS)

    EV = (p_ref * K * (1 - fee) - fixed) / (p_buy * (1 + tax) + ship) - 1,   K = E[exit / p_ref]

so the same player has an EV at the SportsCardsPro price and another at today's eBay ask, and a
break-even ("max pay") price where EV = 0.

    python ev.py validate     # leave-one-year-out calibration of EV vs realised net
    python ev.py score        # current players -> raw/ev_2026.csv
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd

import study_basket as sb
import study_strategies as ss
from load import HERE, REPO, tables
from study_debut import boot

C = sb.COSTS
# the Oct 5 scrape prices every card for 2026-10 (Sep is thin, ffill bridges it), so the 2025 cohort can
# run to its real October exit and keep its September call-ups; set here only, so earlier studies keep
# their documented numbers
sb.LAST_MONTH = pd.Period("2026-10", "M")


def _robust_pop_factory():
    """Raw days 0-3 debut pop per price band from ~200 players (base first autos + every priced
    parallel, 2026-10-24 scrape; debut_pop.day_pops), shrunk toward the pooled mean with 10 pseudo
    players. Replaces study_strategies.pop's 56 base-card estimate for raw; PSA 10 unchanged."""
    orig = ss.pop

    @lru_cache
    def pop(grade: str = "raw"):
        if grade != "raw":
            return orig(grade)
        import debut_pop as dp
        d = dp.day_pops()
        sm = dp.debuts_smooth("raw").set_index("player_id")
        j = d.join(sm[["p_pre"]], how="inner")
        j["band"] = pd.cut(j.p_pre, ss.BANDS, labels=ss.BAND_LBL, right=False)
        m = np.exp(j.day_pop)
        pooled = m.mean()
        g = m.groupby(j.band, observed=True)
        mult = ((g.sum() + 10 * pooled) / (g.size() + 10)).reindex(ss.BAND_LBL).fillna(pooled)
        return mult, pd.DataFrame({"cards": g.size(), "mult": g.mean(), "used": mult})
    return pop


ss.pop = _robust_pop_factory()
PBANDS = [0, 10, 25, 60, 1e9]
PBAND_LBL = ["<$10", "$10-25", "$25-60", "$60+"]
SHRINK = 25          # pseudo-count toward the pooled multiple
NEW100 = 0.05       # P(new Top-100 within a year) that moves the miss path (+9%, ev.py notes)
WINSOR = 0.99        # cap exit multiples at this quantile before averaging (one 9x card is not a rate)


def _level(lv: pd.Series) -> pd.Series:
    return lv.map(lambda l: "AAA" if l == "AAA" else ("AA" if l == "AA" else "low")).fillna("low")


@lru_cache
def history(grade: str = "raw") -> pd.DataFrame:
    b = ss.positions(0, grade)
    lv = pd.concat([sb.top_level(y).rename("level").to_frame().assign(year=y) for y in ss.YEARS])
    b = b.merge(lv.reset_index(), on=["player_id", "year"], how="left")
    b["lvl"] = _level(b.level)
    b["band"] = pd.cut(b.p0, PBANDS, labels=PBAND_LBL, right=False).astype(str)
    b["tier"] = tier(b.p_top100.where(b.elig_top100.astype(bool), 0.0))
    b["mult"] = b.p_exit / b.p0
    # each debuter's own observed pop (monthly, debut_pop.debuts) around his band's mean: the exit then
    # carries player-level variation instead of one band multiple for everyone
    import debut_pop as dp
    po = dp.debuts(grade).set_index("player_id")["pop"].clip(-1.5, 2.5)
    b["pop_obs"] = b.player_id.map(po).where(b.debuted)
    dev = b.pop_obs - b.groupby("band").pop_obs.transform("mean")
    b["p_exit_obs"] = b.p_exit * np.exp(dev.fillna(0.0))
    b["net_obs"] = sb.net(b.p0, b.p_exit_obs)
    b["mult_obs"] = b.p_exit_obs / b.p0
    return b


@lru_cache
def _pop_features(grade: str) -> pd.DataFrame:
    """debut_pop features for every November position, as of January of the season he could debut in"""
    import debut_pop as dp
    b = history(grade)
    f = pd.DataFrame({"player_id": b.player_id, "debut_month": [pd.Period(f"{y + 1}-01", "M") for y in b.year],
                      "p_pre": b.p0, "pop": np.nan, "trend6": np.nan})
    x = dp.features(f.reset_index(drop=True))
    x["p_deb1"], x["p_top100"] = b.p_deb1.to_numpy(), b.p_top100.to_numpy()
    return x


def pop_pred(grade: str, exclude_debut_year: int | None) -> np.ndarray:
    import debut_pop as dp
    return dp.fit_core(grade, exclude_debut_year)(_pop_features(grade))


def tier(p_new_top100: pd.Series) -> pd.Series:
    """likely new Top-100 entrant next list (walk-forward or production P >= NEW100) vs not"""
    return np.where(p_new_top100.fillna(0) >= NEW100, "t100", "-")


def fit(b: pd.DataFrame) -> dict:
    """Shrunk mean exit multiples. Debut path: per price band. Miss path: per band x Top-100 tier.
    Each cell is shrunk toward its band, each band toward the path's pooled mean."""
    cap = b.mult.quantile(WINSOR)
    b = b.assign(m=b.mult.clip(upper=cap), cell=np.where(b.debuted, "-", b.tier))
    out = {}
    for path, g in b.groupby("debuted"):
        pooled = g.m.mean()
        by_band = g.groupby("band").m.agg(["sum", "size"])
        band_m = (by_band["sum"] + SHRINK * pooled) / (by_band["size"] + SHRINK)
        for (band, cell), r in g.groupby(["band", "cell"]).m.agg(["sum", "size"]).iterrows():
            out[(path, band, cell)] = (r["sum"] + SHRINK * band_m.get(band, pooled)) / (r["size"] + SHRINK)
        for band, v in band_m.items():
            out[(path, band, None)] = v
        out[(path, None, None)] = pooled
    return out


def multiples(params: dict, band, tr) -> tuple[np.ndarray, np.ndarray]:
    def look(path, bd, cell):
        return params.get((path, bd, cell), params.get((path, bd, None), params[(path, None, None)]))
    md = np.array([look(True, bd, "-") for bd in band])
    mm = np.array([look(False, bd, t) for bd, t in zip(band, tr)])
    return md, mm


def ev(p: np.ndarray, p_ref: np.ndarray, p_buy: np.ndarray, md: np.ndarray, mm: np.ndarray) -> np.ndarray:
    k = p * md + (1 - p) * mm
    return (p_ref * k * (1 - C["sell_fee"]) - C["sell_fixed"]) / (p_buy * (1 + C["buy_tax"]) + C["buy_ship"]) - 1


def max_pay(p, p_ref, md, mm):
    """Entry price at which EV = 0, given the reference price the multiples apply to."""
    k = p * md + (1 - p) * mm
    return (p_ref * k * (1 - C["sell_fee"]) - C["sell_fixed"] - C["buy_ship"]) / (1 + C["buy_tax"])


def player_debut_mult(md_band: np.ndarray, pred: np.ndarray, band, center: dict) -> np.ndarray:
    """band debut multiple scaled by the player's predicted pop relative to the average debuter of his band"""
    c = np.array([center.get(bd, np.nan) for bd in band])
    return md_band * np.exp(np.nan_to_num(pred - c))


def loyo(grade: str = "raw", pcol: str = "p_deb1", player: bool = False, obs: bool = False) -> pd.DataFrame:
    """obs: fit the multiples on, and score against, exits that carry each debuter's own pop."""
    b = history(grade).copy()
    if obs:
        b["mult"], b["net"] = b.mult_obs, b.net_obs
    out = []
    for y in ss.YEARS:
        trm, tem = (b.year != y).to_numpy(), (b.year == y).to_numpy()
        tr, te = b[trm], b[tem].copy()
        prm = fit(tr)
        md, mm = multiples(prm, te.band, te.tier)
        if player:
            pred = pop_pred(grade, y + 1)
            center = pd.Series(pred[trm][tr.debuted.to_numpy()]).groupby(tr.band[tr.debuted].to_numpy()).mean().to_dict()
            te["pop_pred"] = pred[tem]
            md = player_debut_mult(md, pred[tem], te.band, center)
        te["md"] = md
        te["ev"] = ev(te[pcol].to_numpy(), te.p0.to_numpy(), te.p0.to_numpy(), md, mm)
        out.append(te)
    return pd.concat(out)


def validate_player():
    """Does the player-level debut pop improve the EV? Both versions scored on exits that carry each
    debuter's observed pop."""
    from scipy.stats import spearmanr
    pd.set_option("display.width", 220)
    for grade in ("raw", "psa10"):
        a, p = loyo(grade, obs=True), loyo(grade, player=True, obs=True)
        dd = p[p.debuted & p.pop_obs.notna()]
        print(f"\n===== {grade}: n={len(p)}; among {len(dd)} debuters, rho(pred pop, observed pop) = "
              f"{spearmanr(dd.pop_pred, dd.pop_obs)[0]:.3f}")
        rows = {}
        for nm, d in (("band EV", a), ("player EV", p)):
            rows[f"{nm} rho(ev,net)"] = {y: round(spearmanr(g.ev, g.net)[0], 3) for y, g in d.groupby("year")}
        print(pd.DataFrame(rows).T.to_string())
        rows = {}
        for nm, d in (("band", a), ("player", p)):
            for thr in (0.0, 0.1, 0.2):
                rows[f"{nm} EV>{thr}"] = ss.summary(d[d.ev > thr])
            for k in (25, 50, 100):
                rows[f"{nm} top{k}"] = ss.summary(d.sort_values("ev", ascending=False).groupby("year").head(k // 4))
        print(pd.DataFrame(rows).T.round(3).to_string())
        p["ev_bin"] = pd.cut(p.ev, [-1, -0.2, 0, 0.1, 0.2, 0.4, 10])
        print(p.groupby("ev_bin", observed=True).agg(n=("net", "size"), pred=("ev", "mean"), real=("net", "mean"),
                                                     debut=("debuted", "mean")).round(3).to_string())


def validate():
    pd.set_option("display.width", 220)
    for grade in ("raw", "psa10"):
        b = history(grade)
        prm = fit(b)
        print(f"\n===== {grade}: in-sample multiples (debut path / miss path) =====")
        for nm, sub in (("all years", b), ("2024-25", b[b.year >= 2024])):
            q = fit(sub)
            print(nm, {f"{'D' if k[0] else 'M'} {k[1]} {k[2]}": round(v, 2) for k, v in q.items() if k[1] and k[2]})

        d = loyo(grade)
        print(f"\n----- {grade}: leave-one-year-out EV vs realised net (n={len(d)}) -----")
        bins = [-1, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 10]
        d["ev_bin"] = pd.cut(d.ev, bins)
        rows = {}
        for k, g in d.groupby("ev_bin", observed=True):
            lo, hi = boot(g.net) if len(g) >= 5 else (np.nan, np.nan)
            rows[str(k)] = {"n": len(g), "pred_ev": g.ev.mean(), "real_net": g.net.mean(), "lo": lo, "hi": hi,
                            "median": g.net.median(), "debut": g.debuted.mean(), "med_p0": g.p0.median(),
                            **{f"y{y % 100}": g[g.year == y].net.mean() for y in ss.YEARS}}
        print(pd.DataFrame(rows).T.round(3).to_string())
        from scipy.stats import spearmanr
        print("spearman(ev, net) by year:", {y: round(spearmanr(g.ev, g.net)[0], 3) for y, g in d.groupby("year")})
        print("spearman(P,  net) by year:", {y: round(spearmanr(g.p_deb1, g.net)[0], 3) for y, g in d.groupby("year")})
        # head-to-head: same basket size, chosen by EV vs by P
        rows = {}
        for k in (25, 50, 100):
            for key in ("ev", "p_deb1"):
                pick = d.sort_values(key, ascending=False).groupby("year").head(k // 4)
                rows[f"top{k} by {key}"] = ss.summary(pick)
            pos = d[d.ev > 0]
            rows["all EV>0"] = ss.summary(pos)
            rows["all EV>0.1"] = ss.summary(d[d.ev > 0.1])
        print(pd.DataFrame(rows).T.round(3).to_string())


EBAY = REPO / "runs" / "experiments" / "ev_20261022" / "ebay" / "prices_buylist_latest.csv"
ASK_OVER_SCP = 1.14  # median lowest Buy-It-Now / SportsCardsPro raw estimate, 159 players on 2026-10-22


VARIANT = ("CPS", "BMA")  # 2025 Sapphire variation, Mega Box autos: not the base first auto
ASK_BAND = (0.5, 2.5)     # an ask outside this multiple of the reference is probably another card


def latest_scp(grade: str) -> pd.DataFrame:
    """Current SportsCardsPro price of each player's BASE first auto (variants only as a fallback),
    with the number of completed sales in the last 180 days that back it."""
    t = tables()
    c = t["cards"]
    c = c[(c.kind == "first_auto") & c.player_id.notna()].copy()
    c["variant"] = c.code.fillna("").str.startswith(VARIANT)
    m = t["monthly"][["id", "month", grade]].dropna()
    last = m.sort_values("month").groupby("id").tail(1).set_index("id")[[grade, "month"]]
    s = t["sales"]
    recent = s[(s.grade == grade) & (s.date >= s.date.max() - pd.Timedelta(days=180))].groupby("id").size()
    c = c.join(last, on="id").join(recent.rename("n_sales_180d"), on="id")
    c = c.dropna(subset=[grade])
    c["n_sales_180d"] = c.n_sales_180d.fillna(0).astype(int)
    c = c.sort_values(["player_id", "variant", "n_sales_180d"], ascending=[True, True, False])
    best = c.drop_duplicates("player_id").set_index("player_id")
    return pd.DataFrame({"price": best[grade], "month": best.month.astype(str), "card": best.label,
                         "n_sales_180d": best.n_sales_180d})


@lru_cache
def calibrator(grade: str):
    """Isotonic map from the leave-one-year-out predicted EV to the realised net of those positions:
    the raw two-path EV is too optimistic at the top (+29% predicted, +11% realised)."""
    from sklearn.isotonic import IsotonicRegression
    d = loyo(grade)
    return IsotonicRegression(out_of_bounds="clip", increasing=True).fit(d.ev, d.net)


def outcome_values(grade: str, years=(2013, 2016), cap_q: float = 0.99) -> pd.Series:
    """Mean current price of a base first auto by settled career outcome, cards from `years`
    (players drafted ~2012-15, careers mostly resolved by 2026). Prices capped at cap_q overall so
    one superstar does not set the class mean. Today's dollars, ~10 years after the card."""
    import sqlite3
    from load import DB
    co = pd.read_sql("select player_id, mlb_debut_year, year_established_mlb, year_all_star_once, "
                     "year_major_award from career_outcomes", sqlite3.connect(DB)).set_index("player_id")
    sc = latest_scp(grade)
    cy = tables()["cards"].drop_duplicates("label").set_index("label").card_year
    sc = sc[sc.card.map(cy).between(*years)].join(co, how="inner")
    sc["cls"] = np.select([sc.year_all_star_once.notna() | sc.year_major_award.notna(), sc.year_established_mlb.notna(),
                           sc.mlb_debut_year.notna()], ["star", "established", "debuted"], "never")
    v = sc.price.clip(upper=sc.price.quantile(cap_q)).groupby(sc.cls).mean()
    return v.reindex(["never", "debuted", "established", "star"])


def score():
    """Current EV per player: production P(debut in 2027) and P(new Top-100 within 1y) from
    prod_probs.py, SportsCardsPro reference prices, today's eBay lowest Buy-It-Now as the entry."""
    pp = pd.read_csv(HERE / "raw" / "prod_probs_2026.csv")
    allsc = pd.read_csv(REPO / "runs" / "current" / "buy_lists" / "all_scored.csv")
    final = set(pd.read_csv(REPO / "runs" / "current" / "buy_lists" / "final.csv").player_id)
    d = allsc[["player_id", "name", "bucket", "current_org", "cur_level_2026", "age_at_snap",
               "p_MLB_DEBUT_3y"]].merge(pp.drop(columns=["cur_level_2026", "p_MLB_DEBUT_3y"]), on="player_id", how="left")
    d["on_buy_list"] = d.player_id.isin(final)
    d["p_new_top100_1y"] = d.p_TOP_100_PROSPECT_1y.where(d.year_top_100.isna(), 0.0)
    eb = pd.read_csv(EBAY)
    eb = eb[eb.has_market == 1].set_index("player_id")[["lowest_buynow_price", "price_p25", "n_listings", "lowest_buynow_url"]]
    d = d.join(eb, on="player_id")
    fits = {g: {"all": fit(history(g)), "recent": fit(history(g)[history(g).year >= 2024])} for g in ("raw", "psa10")}
    out = []
    for g in ("raw", "psa10"):
        sc = latest_scp(g).add_prefix(f"scp_{g}_")
        x = d.join(sc, on="player_id")
        ref = x[f"scp_{g}_price"]
        if g == "raw":
            # a SportsCardsPro price with no sale behind it is the site's guess: prefer today's ask
            backed = x[f"scp_{g}_n_sales_180d"].fillna(0) >= 2
            ask = x.lowest_buynow_price
            use_ask = ask.notna() & (ref.isna() | ~backed)
            x["ref_source"] = np.where(use_ask, "ebay ask", np.where(ref.notna(), np.where(backed, "scp", "scp (no sales)"), None))
            ref = ref.where(~use_ask, ask / ASK_OVER_SCP)
            ok = ask.notna() & (ask / ref).between(*ASK_BAND)
            x["ask_flag"] = np.where(ask.notna() & ~ok, "ask looks like another card", "")
            buy = ask.where(ok, ref * ASK_OVER_SCP)
        else:
            buy = ref * ASK_OVER_SCP  # no PSA 10 asks pulled; assume the raw ask premium
        band = pd.cut(ref, PBANDS, labels=PBAND_LBL, right=False).astype(str)
        tr = tier(x.p_new_top100_1y)
        p = x.p_MLB_DEBUT_1y.to_numpy()
        for scen, prm in fits[g].items():
            md, mm = multiples(prm, band, tr)
            x[f"{g}_ev_{scen}"] = ev(p, ref.to_numpy(), buy.to_numpy(), md, mm)
            x[f"{g}_maxpay_{scen}"] = max_pay(p, ref.to_numpy(), md, mm)
            x[f"{g}_ev_fair_{scen}"] = ev(p, ref.to_numpy(), ref.to_numpy(), md, mm)  # paying the sold-price estimate
        x[f"{g}_ev_cal"] = calibrator(g).predict(x[f"{g}_ev_all"].fillna(-1).to_numpy())
        x.loc[x[f"{g}_ev_all"].isna(), f"{g}_ev_cal"] = np.nan
        x[f"{g}_ev_fair_cal"] = calibrator(g).predict(x[f"{g}_ev_fair_all"].fillna(-1).to_numpy())
        x.loc[x[f"{g}_ev_fair_all"].isna(), f"{g}_ev_fair_cal"] = np.nan
        x[f"{g}_ref"] = ref
        x[f"{g}_buy"] = buy
        # player-level debut pop (debut_pop.py): shown, not used in the EV (it did not improve selection)
        import debut_pop as dp
        f = pd.DataFrame({"player_id": x.player_id, "debut_month": pd.Period("2027-01", "M"), "p_pre": ref,
                          "pop": np.nan, "trend6": np.nan}).reset_index(drop=True)
        fx = dp.features(f)
        fx["p_deb1"] = x.p_MLB_DEBUT_1y.to_numpy()
        pred = dp.fit_core(g)(fx)
        hb = history(g)
        hp = pop_pred(g, None)
        center = pd.Series(hp[hb.debuted.to_numpy()]).groupby(hb.band[hb.debuted].to_numpy()).mean()
        md_band, _ = multiples(fits[g]["all"], band, tr)
        x[f"{g}_debut_mult"] = player_debut_mult(md_band, pred, band, center.to_dict())
        x[f"{g}_pop_index"] = np.exp(pred - band.map(center).astype(float).to_numpy())
        out.append(x)
    r = out[0].join(out[1][[c for c in out[1].columns if c.startswith(("psa10_", "scp_psa10_"))]])
    # long run: P(settled career outcome within 6y) x what such a card is worth a decade on
    ps = r.p_STAR_PLUS_ELITE_6y.clip(0, 1)
    pe = (r.p_ESTABLISHED_MLB_6y - ps).clip(0, 1)
    pdb = (r.p_MLB_DEBUT_6y - r.p_ESTABLISHED_MLB_6y).clip(0, 1)
    pn = (1 - ps - pe - pdb).clip(0, 1)
    for g in ("raw", "psa10"):
        v = outcome_values(g)
        r[f"{g}_longrun_value"] = pn * v["never"] + pdb * v["debuted"] + pe * v["established"] + ps * v["star"]
        r[f"{g}_longrun_star_share"] = ps * v["star"] / r[f"{g}_longrun_value"]
    print("outcome values (raw):", outcome_values("raw").round(1).to_dict(), " (psa10):", outcome_values("psa10").round(1).to_dict())
    r = r[r.raw_ref.notna() | r.psa10_ref.notna()]
    r["best_ev"] = r[["raw_ev_all", "psa10_ev_all"]].max(axis=1)
    r["best_copy"] = np.where(r.psa10_ev_all.fillna(-9) > r.raw_ev_all.fillna(-9), "PSA 10", "raw")
    r = r.sort_values("best_ev", ascending=False)
    keep = ["player_id", "name", "on_buy_list", "bucket", "current_org", "cur_level_2026", "age_at_snap",
            "p_MLB_DEBUT_1y", "p_MLB_DEBUT_3y", "p_new_top100_1y", "p_TOP_100_PROSPECT_6y", "p_ESTABLISHED_MLB_6y",
            "p_STAR_PLUS_ELITE_6y", "ref_source", "ask_flag", "scp_raw_card", "scp_raw_price", "scp_raw_n_sales_180d", "lowest_buynow_price",
            "n_listings", "raw_buy", "raw_pop_index", "raw_debut_mult", "raw_ev_fair_all", "raw_ev_fair_recent",
            "raw_ev_fair_cal", "raw_ev_cal", "raw_ev_all", "raw_ev_recent", "raw_maxpay_all", "raw_maxpay_recent",
            "scp_psa10_price", "psa10_buy", "psa10_ev_fair_all", "psa10_ev_fair_recent", "psa10_ev_fair_cal",
            "psa10_ev_cal", "psa10_ev_all", "psa10_ev_recent", "psa10_maxpay_all",
            "psa10_maxpay_recent", "best_copy", "best_ev", "raw_longrun_value", "raw_longrun_star_share",
            "psa10_longrun_value", "p_MLB_DEBUT_6y", "lowest_buynow_url"]
    r[keep].to_csv(HERE / "raw" / "ev_2026.csv", index=False)
    pd.set_option("display.width", 250)
    show = r[r.best_ev > 0][["name", "cur_level_2026", "p_MLB_DEBUT_1y", "p_new_top100_1y", "scp_raw_price",
                             "lowest_buynow_price", "raw_ev_all", "raw_ev_recent", "raw_maxpay_all", "scp_psa10_price",
                             "psa10_ev_all", "psa10_ev_recent", "psa10_maxpay_all", "best_copy"]]
    print(f"{len(r)} priced players; EV>0 (all-years): {len(show)}; EV>0.1: {(r.best_ev > 0.1).sum()}")
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    {"validate": validate, "score": score, "player": validate_player}.get(sys.argv[1] if len(sys.argv) > 1 else "validate", validate)()
