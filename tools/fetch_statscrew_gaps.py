"""Fill the MLB Stats API's pre-1993 minor-league holes from StatsCrew (2026-09-30).

The API has whole-league season lines from 1988, except 1989 (no South Atlantic, NY-Penn,
Northwest or rookie leagues) and 1992 (no Florida State or Carolina League). StatsCrew team pages
carry batting and pitching lines for players with significant playing time (roughly 20 batters and
18 pitchers a team): the missing players are selected by playing time, never by later outcome.

Rows are translated to the API's field names and APPENDED to reference/pre2005/seasons/
<year>_<sportId>_<group>.json with "source": "statscrew", so build_pre2005_ext.py and
build_pre2005_context.py parse them through the production loader unchanged. Player ids: the MLBAM
id of the 1988-1995 draftee with the same normalised name (organisation breaks ties); a line with
no unique match gets a stable negative id and serves as league context only.

Polite: one request every 1.5 s; raw pages cached under reference/pre2005/statscrew/raw/.

    python tools/fetch_statscrew_gaps.py
"""
from __future__ import annotations

import glob
import io
import json
import re
import time
import unicodedata
import zlib
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "reference" / "pre2005"
RAW = PRE / "statscrew" / "raw"
BASE = "https://www.statscrew.com/minorbaseball"
HDR = {"User-Agent": "Mozilla/5.0 (prospect research; low-rate; contact arispiesberger@gmail.com)"}
# (year, StatsCrew league code, MLB sportId, league name)
GAPS = [(1989, "SALL3", 14, "South Atlantic League"), (1989, "NYPL", 15, "New York-Penn League"),
        (1989, "NORW", 15, "Northwest League"), (1989, "APPY", 16, "Appalachian League"),
        (1989, "ARIZ", 16, "Arizona League"), (1989, "GULF", 16, "Gulf Coast League"),
        (1989, "PION", 16, "Pioneer League"),
        (1992, "FLOR", 13, "Florida State League"), (1992, "CARL", 13, "Carolina League")]
BAT = {"GP": "gamesPlayed", "PA": "plateAppearances", "AB": "atBats", "R": "runs", "H": "hits",
       "2B": "doubles", "3B": "triples", "HR": "homeRuns", "RBI": "rbi", "SB": "stolenBases",
       "CS": "caughtStealing", "BB": "baseOnBalls", "SO": "strikeOuts", "HBP": "hitByPitch",
       "SH": "sacBunts", "SF": "sacFlies", "IBB": "intentionalWalks", "DP": "groundIntoDoublePlay",
       "TB": "totalBases", "BA": "avg", "OBP": "obp", "SLG": "slg"}
PIT = {"W": "wins", "L": "losses", "ERA": "era", "G": "gamesPitched", "GS": "gamesStarted",
       "GF": "gamesFinished", "CG": "completeGames", "SHO": "shutouts", "SV": "saves", "H": "hits",
       "R": "runs", "ER": "earnedRuns", "HR": "homeRuns", "BB": "baseOnBalls", "IBB": "intentionalWalks",
       "SO": "strikeOuts", "HBP": "hitBatsmen", "BK": "balks", "WP": "wildPitches", "BF": "battersFaced"}
_last = [0.0]


def get(url: str) -> str:
    f = RAW / (re.sub(r"[^A-Za-z0-9]+", "_", url.split("minorbaseball/")[-1]) + ".html")
    if f.exists():
        return f.read_text(encoding="utf-8")
    wait = 1.5 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers=HDR, timeout=60)
    _last[0] = time.time()
    r.raise_for_status()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(r.text, encoding="utf-8")
    return r.text


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


def ip_api(x) -> str | None:
    """StatsCrew 172.7 = 172 2/3 -> API '172.2'."""
    if pd.isna(x):
        return None
    whole = int(x)
    thirds = int(round((float(x) - whole) * 3))
    if thirds == 3:
        whole, thirds = whole + 1, 0
    return f"{whole}.{thirds}"


def cohort():
    """normalised name -> [(mlbam, draft org name)] for the 1988-1995 draft classes."""
    out: dict[str, list] = {}
    for f in glob.glob(str(PRE / "drafts" / "*.json")):
        if not 1988 <= int(Path(f).stem) <= 1995:
            continue
        for rd in json.loads(Path(f).read_text(encoding="utf-8")).get("drafts", {}).get("rounds", []):
            for k in rd.get("picks", []):
                p = k.get("person") or {}
                if p.get("id"):
                    out.setdefault(norm(p.get("fullName", "")), []).append((p["id"], (k.get("team") or {}).get("name", "")))
    return out


def main():
    coh = cohort()
    stats = {"matched": 0, "context": 0, "ambiguous": 0}
    for year, code, sid, lname in GAPS:
        lp = get(f"{BASE}/l-{code}/y-{year}")
        teams = sorted(set(re.findall(rf"minorbaseball/(?:stats|roster)/(t-[a-z0-9]+)/y-{year}", lp)))
        add = {"hitting": [], "pitching": []}
        for t in teams:
            page = get(f"{BASE}/stats/{t}/y-{year}")
            m = re.search(r"Affiliation:\s*(?:<[^>]+>\s*)*([^<]+)<", page)
            org = (m.group(1).strip() if m else "")
            tname = re.search(r"<title>\s*\d{4}\s+(.*?)\s+(?:minor league|Statistics)", page)
            tname = tname.group(1) if tname else t
            for tb in pd.read_html(io.StringIO(page)):
                cols = set(tb.columns)
                grp = "pitching" if {"ERA", "IP"} <= cols else "hitting" if {"AB", "PA"} <= cols else None
                if grp is None:
                    continue
                for _, r in tb.iterrows():
                    nm = str(r["Player"]).strip()
                    if not nm or nm.lower().startswith(("total", "team")):
                        continue
                    cands = coh.get(norm(nm), [])
                    if len(cands) > 1:
                        cands = [c for c in cands if org and org.split()[-1] in c[1]] or cands
                    if len(cands) == 1:
                        pid = cands[0][0]
                        stats["matched"] += 1
                    else:
                        stats["ambiguous" if len(cands) > 1 else "context"] += 1
                        pid = -(zlib.crc32(f"{nm}|{t}|{year}".encode()) % 10**9) - 1
                    mp = PIT if grp == "pitching" else BAT
                    stat = {v: (None if pd.isna(r.get(k)) else (float(r[k]) if k in ("ERA", "BA", "OBP", "SLG") else int(r[k])))
                            for k, v in mp.items() if k in r.index}
                    if grp == "pitching":
                        stat["inningsPitched"] = ip_api(r.get("IP"))
                        ip = r.get("IP")
                        stat["outs"] = None if pd.isna(ip) else int(int(ip) * 3 + round((float(ip) - int(ip)) * 3))
                    first, _, last = nm.partition(" ")
                    add[grp].append({"season": str(year), "stat": stat, "source": "statscrew",
                                     "team": {"id": None, "name": tname, "org": org},
                                     "player": {"id": pid, "fullName": nm, "firstName": first, "lastName": last},
                                     "league": {"id": None, "name": lname},
                                     "position": {"abbreviation": "P" if grp == "pitching" else ""}})
        for grp, rows in add.items():
            f = PRE / "seasons" / f"{year}_{sid}_{grp}.json"
            cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
            cur = [s for s in cur if not (s.get("source") == "statscrew" and (s.get("league") or {}).get("name") == lname)]
            f.write_text(json.dumps(cur + rows), encoding="utf-8")
        print(f"[statscrew] {year} {lname}: {len(teams)} teams, {len(add['hitting'])} batting + "
              f"{len(add['pitching'])} pitching lines", flush=True)
    print(f"[statscrew] cohort matches {stats['matched']:,}; context-only {stats['context']:,}; "
          f"ambiguous {stats['ambiguous']:,}")


if __name__ == "__main__":
    main()
