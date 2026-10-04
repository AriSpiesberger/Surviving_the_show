"""Turn the raw SportsCardsPro scrape into tidy tables joined to prospects.db.

    cards    one row per scraped card (product id, set, card code, kind, player_id)
    monthly  card x month raw / PSA 9 / PSA 10 price (USD), the SportsCardsPro
             monthly "loose" price series
    sales    individual completed sales (date, price, title) per card and grade
    people   player_id -> MLB debut date, birth date, draft info

Card kinds:
    first_auto   the 1st Bowman Chrome prospect / draft-pick autograph (CPA, CDA,
                 BCAP, BDPA ...), the card the buy list trades
    rookie_auto  Bowman Chrome rookie autograph (CRA, BCAR ...) issued once in MLB
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "raw"
REPO = HERE.parent.parent
DB = REPO / "prospects.db"


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"[^a-z]+", " ", s).strip()


_CODE = re.compile(r"#\s*([A-Z0-9]+)-")
_ROOKIE = re.compile(r"rookie", re.I)


def card_kind(set_name: str) -> str:
    if _ROOKIE.search(set_name):
        return "rookie_auto"
    if re.search(r"prospect|draft|picks", set_name, re.I):
        return "first_auto"
    return "other"


def card_year(set_name: str) -> int | None:
    m = re.search(r"(20\d\d)", set_name)
    return int(m.group(1)) if m else None


def _jsonl(kind: str) -> list[dict]:
    p = RAW / f"{kind}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


@lru_cache
def people() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    p = pd.read_sql(
        """select p.player_id, p.mlbam_id, p.name, p.is_pitcher, p.birth_date,
                  p.draft_year, p.draft_round, p.draft_pick, p.is_international,
                  c.mlb_debut_year, c.career_war, c.all_star_selections,
                  c.year_established_mlb, c.year_all_star_once, c.year_top_100
           from prospects p left join career_outcomes c using(player_id)""", con)
    mlb = pd.read_json(RAW / "mlb_people.jsonl", lines=True)
    mlb = mlb.rename(columns={"id": "mlbam_id"})[["mlbam_id", "mlbDebutDate", "lastPlayedDate"]]
    p["mlbam_id"] = pd.to_numeric(p.mlbam_id, errors="coerce")
    p = p.merge(mlb, on="mlbam_id", how="left")
    p["debut_date"] = pd.to_datetime(p.mlbDebutDate)
    p["norm"] = p.name.map(norm_name)
    return p


def _match_players(cards: pd.DataFrame) -> pd.Series:
    """Card -> player_id by normalized name. A name shared by several prospects is
    broken toward the one whose draft/entry year is closest to (and not after) the
    card year; anything still ambiguous is left unmatched."""
    p = people()
    p = p.assign(entry=p.draft_year.fillna(p.mlb_debut_year))
    by = {k: g for k, g in p.groupby("norm")}
    out = []
    for name, yr in zip(cards.norm, cards.card_year):
        g = by.get(name)
        if g is None:
            out.append(None)
        elif len(g) == 1:
            out.append(g.player_id.iloc[0])
        else:
            gap = (yr - g.entry).where(yr >= g.entry - 1)
            gap = gap.dropna()
            out.append(g.loc[gap.idxmin(), "player_id"] if len(gap) and (gap == gap.min()).sum() == 1 else None)
    return pd.Series(out, index=cards.index)


@lru_cache
def tables() -> dict[str, pd.DataFrame]:
    prods = [r for r in _jsonl("product") if "id" in r]  # drop early smoke-test rows
    rows, monthly, sales = [], [], []
    for r in prods:
        name = re.sub(r"\[.*?\]", "", r["n"]).split("#")[0].strip()
        code = (_CODE.search(r["n"]) or [None, None])[1]
        rows.append({"id": r["id"], "set": r["c"].replace("Baseball Cards ", ""), "card_name": name,
                     "code": code, "label": r["n"]})
        for grade, key in (("raw", "used"), ("psa9", "graded"), ("psa10", "manualonly")):
            for ym, cents in (r.get("chart") or {}).get(key, []):
                monthly.append((r["id"], ym, grade, cents / 100))
        for grade, lst in (r.get("sales") or {}).items():
            for d, price, title in lst:
                sales.append((r["id"], d, grade, price, title))
    cards = pd.DataFrame(rows).drop_duplicates("id")
    cards["card_year"] = cards.set.map(card_year)
    cards["kind"] = cards.set.map(card_kind)
    cards["norm"] = cards.card_name.map(norm_name)
    cards["player_id"] = _match_players(cards)
    m = pd.DataFrame(monthly, columns=["id", "month", "grade", "price"])
    m["month"] = pd.PeriodIndex(m.month, freq="M")
    m = m.pivot_table(index=["id", "month"], columns="grade", values="price").reset_index()
    s = pd.DataFrame(sales, columns=["id", "date", "grade", "price", "title"]).drop_duplicates()
    s["date"] = pd.to_datetime(s.date)
    return {"cards": cards, "monthly": m, "sales": s, "people": people()}


if __name__ == "__main__":
    t = tables()
    c = t["cards"]
    print(c.groupby("kind").agg(n=("id", "size"), matched=("player_id", lambda x: x.notna().sum())))
    print(c.groupby("card_year").size())
    print(t["monthly"].groupby(t["monthly"].month.dt.year).size())
    print(len(t["sales"]), "sales")
