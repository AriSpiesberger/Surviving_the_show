"""Unify FanGraphs-Board + Trouble-With-The-Curve into point-in-time tables
keyed (player_id, season, source) -- mapping EVERYTHING usable.

Outputs:
  scratch/fangraphs_board/scouting_grades_pointintime.csv   numeric + encoded
  scratch/fangraphs_board/scouting_reports_text.csv         report text (for
                                                            the embedding step)

FG board (2017-2026): present/future tool grades (incl fArm), pitch grades,
velocity (Vel/Touch/Range), spin (bRPM/fRPM), physical (Frame/Athleticism/
Performer), ranks, ETA, fantasy, service time, ordinal Risk/Variance/Levers/
Trend, encoded Bats/Throws/Player_Type/Signed_Mkt/School_Type/Contact_Style/
FBType/FYPD, parsed Versatility(count) + Dist_Raw(5 comps), TJ surgery date.
TWTC (2013-2019): single grades -> FUTURE cols (+ arm/age/eta).
Pure ids/urls (ID/UPID/URL/route/YouTube/Agent/school-name/geo) are dropped.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FG_PARSED = REPO_ROOT / "scratch" / "fangraphs_board" / "parsed"
XWALK = REPO_ROOT / "scratch" / "fangraphs_board" / "fg_crosswalk.csv"
TWTC = REPO_ROOT / "scratch" / "external" / "twtc" / "data" / "twtc.csv"
DB = REPO_ROOT / "prospects_snapshot.db"
OUT = REPO_ROOT / "scratch" / "fangraphs_board" / "scouting_grades_pointintime.csv"
OUT_TXT = REPO_ROOT / "scratch" / "fangraphs_board" / "scouting_reports_text.csv"

FG_NUMERIC = {
    "pHit": "hit_p", "fHit": "hit_f", "pGame": "gamepower_p", "fGame": "gamepower_f",
    "pRaw": "rawpower_p", "fRaw": "rawpower_f", "pSpd": "speed_p", "fSpd": "speed_f",
    "pFld": "field_p", "fFld": "field_f", "pArm": "arm_p", "fArm": "arm_f",
    "pFB": "fastball_p", "fFB": "fastball_f", "pSL": "slider_p", "fSL": "slider_f",
    "pCB": "curve_p", "fCB": "curve_f", "pCH": "change_p", "fCH": "change_f",
    "pCT": "cutter_p", "fCT": "cutter_f", "pSPL": "splitter_p", "fSPL": "splitter_f",
    "pCMD": "command_p", "fCMD": "command_f",
    "Pitch_Sel": "pitch_sel", "Bat_Ctrl": "bat_ctrl",
    "FV_Current": "fv", "Ovr_Rank": "ovr_rank", "Org_Rank": "org_rank",
    "cOVR": "ovr_rank_c", "cORG": "org_rank_c",
    "ETA_Current": "eta", "Age": "age", "Height": "height", "Weight": "weight",
    "Athleticism": "athleticism", "Frame": "frame", "Performer": "performer",
    "HardHit%": "hardhit", "Vel": "velo", "Touch": "velo_touch",
    "Delivery": "delivery", "bRPM": "spin_break", "fRPM": "spin_fastball",
    "Amateur_Rk": "amateur_rk", "Class_Rk": "class_rk", "Draft_Rnd": "draft_rnd",
    "Signed_Yr": "signed_yr", "Sign_Bonus": "sign_bonus",
    "Fantasy_Redraft": "fantasy_redraft", "Fantasy_Dynasty": "fantasy_dynasty",
    "servicetime": "servicetime",
}
FG_NOMINAL = {  # encoded to stable integer codes (computed post-concat)
    "Bats": "bats", "Throws": "throws", "Player_Type": "player_type",
    "Signed_Mkt": "signed_mkt", "School_Type": "school_type",
    "Contact_Style": "contact_style", "FBType": "fb_type",
    "FYPD_Eligible": "fypd_eligible",
}
RISK_MAP = {"low": 1, "med": 2, "medium": 2, "short": 2, "high": 3,
            "very high": 4, "extreme": 4}
LEVERS_MAP = {"short": 1, "med": 2, "medium": 2, "long": 3}
TREND_MAP = {"&uarr;": 1, "↑": 1, "&darr;": -1, "↓": -1}

# assembled after we know the parsed/encoded names
PARSED = ["risk", "variance", "levers", "trend", "velo_lo", "velo_hi",
          "had_tj", "tj_year", "versatility_count",
          "dist_1", "dist_2", "dist_3", "dist_4", "dist_5"]
FEATURES = list(dict.fromkeys(list(FG_NUMERIC.values())
                              + list(FG_NOMINAL.values()) + PARSED))
OUT_COLS = ["player_id", "season", "source"] + FEATURES


def norm_name(s) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return "".join(t for t in re.findall(r"[a-z]+", s)
                   if t not in {"jr", "sr", "ii", "iii", "iv", "v"})


def gnum(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns \
        else pd.Series(np.nan, index=df.index)


def gcode(df, col):
    """Stable integer code for a nominal column; NaN stays NaN."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    cc = df[col].astype("category").cat.codes.astype(float)
    return cc.where(cc >= 0)


def build_fg():
    xw = pd.read_csv(XWALK)
    pid_map = dict(zip(xw.PlayerId.astype(str), xw.player_id))
    df = pd.concat([pd.read_csv(p, low_memory=False)
                    for p in sorted(FG_PARSED.glob("*.csv"))], ignore_index=True)
    df["player_id"] = df["PlayerId"].astype(str).map(pid_map)
    df = df[df.player_id.notna()].reset_index(drop=True)

    out = {"player_id": df.player_id, "season": gnum(df, "Season"),
           "source": "fg_board"}
    for src, dst in FG_NUMERIC.items():
        out[dst] = gnum(df, src)
    for src, dst in FG_NOMINAL.items():
        out[dst] = gcode(df, src)
    low = lambda c: (df[c].astype(str).str.strip().str.lower()
                     if c in df.columns else pd.Series(np.nan, index=df.index))
    out["risk"] = low("cRisk").map(RISK_MAP)
    out["variance"] = low("Variance").map(RISK_MAP)
    out["levers"] = low("Levers").map(LEVERS_MAP)
    out["trend"] = (df["Trend"].astype(str).str.strip().map(TREND_MAP)
                    if "Trend" in df.columns else np.nan)
    rng = (df["Range"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
           if "Range" in df.columns else pd.DataFrame(index=df.index, columns=[0, 1]))
    out["velo_lo"], out["velo_hi"] = gnum(rng, 0), gnum(rng, 1)
    if "TJDate" in df.columns:
        out["had_tj"] = df["TJDate"].notna().astype(int)
        out["tj_year"] = pd.to_datetime(df["TJDate"], errors="coerce").dt.year
    else:
        out["had_tj"], out["tj_year"] = 0, np.nan
    out["versatility_count"] = (df["Versatility"].astype(str).str.count("/") + 1
                                ).where(df["Versatility"].notna()) \
        if "Versatility" in df.columns else np.nan
    if "Dist_Raw" in df.columns:
        d = df["Dist_Raw"].astype(str).str.split(":", expand=True)
        for i in range(5):
            out[f"dist_{i+1}"] = gnum(d, i) if i in d.columns else np.nan
    else:
        for i in range(5):
            out[f"dist_{i+1}"] = np.nan
    feats = pd.DataFrame(out)
    text = pd.DataFrame({
        "player_id": df.player_id, "season": gnum(df, "Season"), "source": "fg_board",
        "summary": df.get("Summary"), "tldr": df.get("TLDR"),
        "ovr_summary": df.get("Ovr_Summary"),
    })
    print(f"FG board: {len(feats):,} rows, {feats.player_id.nunique():,} players")
    return feats, text


def build_twtc(mlbam2pid, namebday2pid):
    df = pd.read_csv(TWTC, low_memory=False)
    df["name"] = df["name"].astype(str).str.lstrip("*").str.strip()
    km = pd.to_numeric(df["key_mlbam"], errors="coerce")
    df["player_id"] = km.where(km > 0).dropna().astype(int).astype(str).map(mlbam2pid)
    need = df.player_id.isna()
    bkey = pd.to_datetime(df["birthdate"], errors="coerce").dt.strftime("%Y-%m-%d")
    bkey = bkey.where(bkey != "1899-12-30")
    fb = pd.Series([namebday2pid.get((norm_name(n), b)) if pd.notna(b) else None
                    for n, b in zip(df["name"], bkey)], index=df.index)
    df.loc[need, "player_id"] = fb[need]
    n_fb = int(need.sum() - df.player_id.isna().sum())
    df = df[df.player_id.notna()].reset_index(drop=True)

    g = lambda c: pd.to_numeric(df.get(c), errors="coerce").replace(0, np.nan)
    out = {f: np.nan for f in FEATURES}
    out.update({"player_id": df.player_id, "season": gnum(df, "year"),
                "source": "twtc", "hit_f": g("Hit"), "gamepower_f": g("Power"),
                "speed_f": g("Run"), "field_f": g("Field"), "arm_f": g("Arm"),
                "fastball_f": g("Fastball"), "slider_f": g("Slider"),
                "curve_f": g("Curveball"), "change_f": g("Changeup"),
                "cutter_f": g("Cutter"), "splitter_f": g("Splitter"),
                "command_f": g("Control"), "age": g("age"), "eta": g("eta"),
                "had_tj": 0})
    feats = pd.DataFrame(out)
    text = pd.DataFrame({"player_id": df.player_id, "season": gnum(df, "year"),
                         "source": "twtc", "summary": df.get("report"),
                         "tldr": np.nan, "ovr_summary": df.get("text")})
    print(f"TWTC: {len(feats):,} rows, {feats.player_id.nunique():,} players "
          f"[{n_fb} via name+DOB fallback]")
    return feats, text


def main():
    c = sqlite3.connect(DB)
    p = pd.read_sql("SELECT player_id,name,birth_date,mlbam_id FROM prospects", c)
    c.close()
    m = p.dropna(subset=["mlbam_id"])
    mlbam2pid = dict(zip(m.mlbam_id.astype(str), m.player_id))
    p["_nm"] = p["name"].map(norm_name)
    p["_bkey"] = pd.to_datetime(p["birth_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    namebday2pid = (p.dropna(subset=["_bkey"]).drop_duplicates(["_nm", "_bkey"])
                    .set_index(["_nm", "_bkey"])["player_id"].to_dict())

    fg, fg_txt = build_fg()
    tw, tw_txt = build_twtc(mlbam2pid, namebday2pid)
    allg = (pd.concat([fg, tw], ignore_index=True)[OUT_COLS]
            .dropna(subset=["season"]))
    allg["season"] = allg["season"].astype(int)
    allg.to_csv(OUT, index=False)

    txt = pd.concat([fg_txt, tw_txt], ignore_index=True).dropna(subset=["season"])
    txt["season"] = txt["season"].astype(int)
    txt.to_csv(OUT_TXT, index=False)

    print(f"\n{'='*60}\nGRADES: {len(allg):,} rows, {allg.player_id.nunique():,} "
          f"players, {allg.season.min()}-{allg.season.max()}, {len(FEATURES)} features")
    print(f"TEXT:   {len(txt):,} rows  ({txt.summary.notna().sum():,} with summary, "
          f"{txt.ovr_summary.notna().sum():,} with long text)")
    cov = allg[FEATURES].notna().mean().sort_values(ascending=False)
    print(f"feature coverage range: {cov.min():.0%} - {cov.max():.0%}; "
          f"median {cov.median():.0%}")
    print(f"Wrote {OUT.name} + {OUT_TXT.name}")


if __name__ == "__main__":
    main()
