"""Build a FanGraphs-Board -> repo player_id crosswalk.

FG board rows carry NO mlbam id — only FG's own PlayerId (numeric for players
with an FG page, or 'sa....' for unsigned amateurs), plus name / birthdate /
position / org. The repo keys on player_id and has mlbam_id (100% populated),
name, birth_date, draft_year, primary_position.

We match in priority order (highest-confidence first); each FG player resolves
to at most one repo player_id, earlier passes win, and one-to-many collisions
are flagged rather than silently picked.

  A. chadwick      numeric FG PlayerId -> Chadwick key_fangraphs -> key_mlbam
                   -> repo.mlbam_id            (exact id; needs pybaseball)
  B. name+bdate    norm(name) + full birth date
  C. name+byear    norm(name) + birth year     (ties -> draft_year/pos)
  D. last+bdate+fi norm(last) + birth date + first initial (nickname variants)
  E. fuzzy         rapidfuzz on name within birth-year bucket (>=92), flagged

Outputs (scratch/fangraphs_board/):
  fg_crosswalk.csv   PlayerId, player_id, mlbam_id, method, confidence, names...
  fg_unmatched.csv   the FG players we could not place (with their seasons)

Usage:
    python -m scripts.build_fg_crosswalk
    python -m scripts.build_fg_crosswalk --no-chadwick   # skip the FG-id bridge
"""
from __future__ import annotations

import argparse
import glob
import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PARSED = REPO_ROOT / "scratch" / "fangraphs_board" / "parsed"
OUT_DIR = REPO_ROOT / "scratch" / "fangraphs_board"
DB = REPO_ROOT / "prospects_snapshot.db"
CHADWICK_CACHE = OUT_DIR / "chadwick_register.csv"

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(s) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    toks = re.findall(r"[a-z]+", s)
    toks = [t for t in toks if t not in _SUFFIXES]
    return "".join(toks)


def norm_last(s) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    toks = [t for t in re.findall(r"[a-z]+", s.lower()) if t not in _SUFFIXES]
    return toks[-1] if toks else ""


def first_initial(s) -> str:
    n = norm_name(s)
    return n[0] if n else ""


# ----------------------------------------------------------------------------

def load_fg() -> pd.DataFrame:
    """Unique FG players (by PlayerId), bio taken from the most recent season,
    plus the list of seasons each appeared in."""
    files = sorted(glob.glob(str(PARSED / "*.csv")))
    if not files:
        raise SystemExit(f"No FG parsed CSVs under {PARSED} — run the scraper.")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
    fg = pd.concat(frames, ignore_index=True)
    # full name: prefer FirstName+LastName, else playerName
    fn = fg.get("FirstName", pd.Series([""] * len(fg)))
    ln = fg.get("LastName", pd.Series([""] * len(fg)))
    combo = (fn.fillna("").astype(str) + " " + ln.fillna("").astype(str)).str.strip()
    fg["_fullname"] = combo.where(combo.str.len() > 1,
                                  fg.get("playerName", "")).fillna("")
    # FG BirthDate is an Excel serial (base 1899-12-30); 0 = missing sentinel.
    bd_raw = pd.to_numeric(fg.get("BirthDate"), errors="coerce")
    bd_raw = bd_raw.where(bd_raw > 1)
    fg["_bdate"] = pd.to_datetime(bd_raw, unit="D", origin="1899-12-30",
                                  errors="coerce")
    seasons = (fg.groupby("PlayerId")["Season"]
               .agg(lambda s: ",".join(sorted({str(int(x)) for x in s
                                                if pd.notna(x)}))))
    # most-recent-season bio per PlayerId
    fg = fg.sort_values("Season")
    uniq = fg.groupby("PlayerId", as_index=False).last()
    uniq = uniq.merge(seasons.rename("_seasons"), on="PlayerId")
    uniq["_nm"] = uniq["_fullname"].map(norm_name)
    uniq["_last"] = uniq["LastName"].map(norm_last) if "LastName" in uniq else \
        uniq["_fullname"].map(norm_last)
    uniq["_fi"] = uniq["_fullname"].map(first_initial)
    uniq["_byear"] = uniq["_bdate"].dt.year
    # Fallback for years with no BirthDate (2017-19): birth year ~ Season - Age.
    age = pd.to_numeric(uniq.get("Age"), errors="coerce")
    season = pd.to_numeric(uniq.get("Season"), errors="coerce")
    approx = (season - age.round())
    uniq["_byear"] = uniq["_byear"].fillna(approx).astype("Int64")
    uniq["_pid_is_num"] = uniq["PlayerId"].astype(str).str.fullmatch(r"\d+")
    return uniq


def load_repo() -> pd.DataFrame:
    c = sqlite3.connect(DB)
    p = pd.read_sql("SELECT player_id,name,birth_date,mlbam_id,draft_year,"
                    "primary_position FROM prospects", c)
    c.close()
    p["_nm"] = p["name"].map(norm_name)
    p["_last"] = p["name"].map(norm_last)
    p["_fi"] = p["name"].map(first_initial)
    p["_bdate"] = pd.to_datetime(p["birth_date"], errors="coerce")
    p["_byear"] = p["_bdate"].dt.year
    p["_bkey"] = p["_bdate"].dt.strftime("%Y-%m-%d")
    return p


def load_chadwick() -> pd.DataFrame | None:
    if CHADWICK_CACHE.exists():
        return pd.read_csv(CHADWICK_CACHE, low_memory=False)
    try:
        from pybaseball import chadwick_register
    except Exception as e:
        print(f"  [chadwick] pybaseball unavailable ({e}); skipping pass A")
        return None
    try:
        reg = chadwick_register()
    except Exception as e:
        print(f"  [chadwick] download failed ({e}); skipping pass A")
        return None
    reg = reg[["key_fangraphs", "key_mlbam", "name_first", "name_last"]].copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reg.to_csv(CHADWICK_CACHE, index=False)
    return reg


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chadwick", action="store_true")
    ap.add_argument("--fuzzy-threshold", type=int, default=92)
    args = ap.parse_args()

    fg = load_fg()
    repo = load_repo()
    print(f"FG unique players: {len(fg):,}  ({fg._pid_is_num.sum():,} numeric "
          f"PlayerId, {(~fg._pid_is_num).sum():,} 'sa' amateurs)")
    print(f"repo prospects:    {len(repo):,}  "
          f"({repo._bdate.notna().sum():,} with birthdate)\n")

    matched: dict[str, dict] = {}   # PlayerId -> match record
    repo_by_mlbam = (repo.dropna(subset=["mlbam_id"])
                     .assign(mlbam_id=lambda d: d.mlbam_id.astype(str))
                     .drop_duplicates("mlbam_id")
                     .set_index("mlbam_id", drop=False))

    def claim(pid, rrow, method, conf):
        if pid in matched:
            return
        matched[pid] = {
            "PlayerId": pid, "player_id": rrow["player_id"],
            "mlbam_id": rrow["mlbam_id"], "method": method, "confidence": conf,
            "fg_name": fgrow_name.get(pid, ""), "repo_name": rrow["name"],
        }

    fgrow_name = dict(zip(fg.PlayerId, fg._fullname))

    # ---- Pass A: Chadwick FG-id bridge ----
    if not args.no_chadwick:
        reg = load_chadwick()
        if reg is not None:
            reg = reg.dropna(subset=["key_fangraphs", "key_mlbam"]).copy()
            reg["key_fangraphs"] = reg["key_fangraphs"].astype("Int64").astype(str)
            reg["key_mlbam"] = reg["key_mlbam"].astype("Int64").astype(str)
            fg2mlbam = dict(zip(reg.key_fangraphs, reg.key_mlbam))
            n = 0
            for _, r in fg[fg._pid_is_num].iterrows():
                pid = str(r.PlayerId)
                mlbam = fg2mlbam.get(pid)
                if mlbam is not None and mlbam in repo_by_mlbam.index:
                    rr = repo_by_mlbam.loc[mlbam]
                    if isinstance(rr, pd.DataFrame):
                        rr = rr.iloc[0]
                    claim(pid, rr, "A_chadwick", "exact")
                    n += 1
            print(f"Pass A (chadwick id):     +{n:,}")

    # helper to run a keyed pass over unmatched FG players
    def keyed_pass(name, conf, fg_key_cols, repo_key_cols, valid):
        rsub = repo.dropna(subset=repo_key_cols)
        # unique repo key -> single candidate; ambiguous keys handled by caller
        grp = rsub.groupby(repo_key_cols)
        lookup = {k: g for k, g in grp}
        n = ambig = 0
        for _, r in fg.iterrows():
            pid = str(r.PlayerId)
            if pid in matched or not valid(r):
                continue
            key = tuple(r[c] for c in fg_key_cols)
            key = key[0] if len(key) == 1 else key
            cand = lookup.get(key)
            if cand is None:
                continue
            if len(cand) == 1:
                claim(pid, cand.iloc[0], name, conf)
                n += 1
            else:
                # tie-break: draft_year match, else position match
                c2 = cand
                if pd.notna(r.get("Signed_Yr")):
                    c2b = cand[cand.draft_year == r.get("Signed_Yr")]
                    if len(c2b) == 1:
                        c2 = c2b
                if len(c2) == 1:
                    claim(pid, c2.iloc[0], name + "+tiebreak", conf)
                    n += 1
                else:
                    ambig += 1
        print(f"Pass {name}: +{n:,}" + (f"  ({ambig} ambiguous skipped)" if ambig else ""))

    # ---- Pass B: name + full birthdate ----
    fg["_bkey"] = fg["_bdate"].dt.strftime("%Y-%m-%d")
    keyed_pass("B_name+bdate", "high", ["_nm", "_bkey"], ["_nm", "_bkey"],
               valid=lambda r: bool(r._nm) and pd.notna(r._bkey))

    # ---- Pass C: name + birth year ----
    keyed_pass("C_name+byear", "med", ["_nm", "_byear"], ["_nm", "_byear"],
               valid=lambda r: bool(r._nm) and pd.notna(r._byear))

    # ---- Pass D: last + birthdate + first initial ----
    keyed_pass("D_last+bdate+fi", "med", ["_last", "_bkey", "_fi"],
               ["_last", "_bkey", "_fi"],
               valid=lambda r: bool(r._last) and pd.notna(r._bkey))

    # ---- Pass F: name + birth-year +/-1, unique repo candidate only ----
    # Recovers Age-fallback off-by-one cases (2017-18 boards lacked BirthDate)
    # while staying safe on common names (require a single repo player_id).
    rsub = repo.dropna(subset=["_byear"])
    by_name = {nm: g for nm, g in rsub.groupby("_nm")}
    nF = 0
    for _, r in fg.iterrows():
        pid = str(r.PlayerId)
        if pid in matched or not r._nm or pd.isna(r._byear):
            continue
        cand = by_name.get(r._nm)
        if cand is None:
            continue
        close = cand[(cand._byear - r._byear).abs() <= 1]
        if close.player_id.nunique() == 1:
            claim(pid, close.iloc[0], "F_name+byear_pm1", "med")
            nF += 1
    print(f"Pass F (name+byear+/-1):  +{nF:,}")

    # ---- Pass E: fuzzy name within birth-year bucket ----
    try:
        from rapidfuzz import fuzz, process
        repo_by_year = {y: g for y, g in repo.dropna(subset=["_byear"]).groupby("_byear")}
        n = 0
        for _, r in fg.iterrows():
            pid = str(r.PlayerId)
            if pid in matched or not r._nm or pd.isna(r._byear):
                continue
            bucket = repo_by_year.get(r._byear)
            if bucket is None or bucket.empty:
                continue
            choices = bucket["_nm"].tolist()
            hit = process.extractOne(r._nm, choices, scorer=fuzz.ratio)
            if hit and hit[1] >= args.fuzzy_threshold:
                claim(pid, bucket.iloc[hit[2]], "E_fuzzy", f"low({hit[1]})")
                n += 1
        print(f"Pass E (fuzzy):           +{n:,}")
    except ImportError:
        print("Pass E skipped (rapidfuzz not installed: pip install rapidfuzz)")

    # ---- assemble + report ----
    xwalk = pd.DataFrame(matched.values())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xwalk.to_csv(OUT_DIR / "fg_crosswalk.csv", index=False)
    unmatched = fg[~fg.PlayerId.astype(str).isin(set(matched))][
        ["PlayerId", "_fullname", "_byear", "Position", "_seasons", "_pid_is_num"]]
    unmatched.to_csv(OUT_DIR / "fg_unmatched.csv", index=False)

    tot = len(fg)
    print(f"\n{'='*60}\nMATCHED {len(xwalk):,}/{tot:,} ({100*len(xwalk)/tot:.1f}%)"
          f"  | unmatched {len(unmatched):,}")
    if len(xwalk):
        print(xwalk["method"].value_counts().to_string())
        # collision check: one repo player claimed by multiple FG players
        dup = xwalk["player_id"].value_counts()
        dup = dup[dup > 1]
        if len(dup):
            print(f"\n[WARN] {len(dup)} repo player_ids claimed by >1 FG player "
                  f"(see fg_crosswalk.csv); review.")
    # unmatched breakdown: numeric-id (real misses) vs 'sa' amateurs
    um_num = unmatched._pid_is_num.sum()
    print(f"\nUnmatched: {um_num:,} numeric-id (should mostly match — review), "
          f"{len(unmatched)-um_num:,} 'sa' amateurs (often not in repo yet)")
    print("Sample unmatched numeric-id players:")
    print(unmatched[unmatched._pid_is_num].head(12)[
        ["PlayerId", "_fullname", "_byear", "_seasons"]].to_string(index=False))


if __name__ == "__main__":
    main()
