"""
prospects/ingestion/backfills/mlb_lahman_seasons.py
==========================================

Pull per-year MLB stats from Lahman and write SeasonStats rows
(level="MLB") for any prospect that matches a Lahman bbrefID.

This is fast (Lahman is local) and complements the MiLB pull, so a player's
random-year feature window can include MLB years if they had any.

Schema mapping (Lahman batting -> SeasonStats):
    PA   = AB + BB + HBP + SF
    AVG  = H / AB
    OBP  = (H + BB + HBP) / PA
    SLG  = TB / AB,  TB = H + 2B + 2*3B + 3*HR
    ISO  = SLG - AVG
    BB%  = BB / PA
    K%   = SO / PA

Lahman pitching -> SeasonStats:
    IP   = IPouts / 3
    ERA  = ERA from table
    K/9  = 9 * SO / IP
    BB/9 = 9 * BB / IP
    HR/9 = 9 * HR / IP
    WHIP = (BB + H) / IP
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np

from prospects.schema import SeasonStats
from prospects.storage import ProspectDB


def _safe_div(num, den):
    try:
        n = float(num); d = float(den)
    except (TypeError, ValueError):
        return None
    return n / d if d > 0 else None


def _normalize_name(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", " ", s).lower().strip()
    return re.sub(r"\s+", " ", s)


def _split(full: str) -> tuple[str, str]:
    parts = _normalize_name(full).split()
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    while parts and parts[-1] in suffixes:
        parts.pop()
    if len(parts) <= 1:
        return ("", parts[0] if parts else "")
    return parts[0], parts[-1]


def pull_mlb_seasons_from_lahman(
    db: ProspectDB,
    csv_path: Optional[str] = "mlb_season_stats.csv",
    verbose: bool = True,
) -> int:
    """
    Walk every prospect, resolve to bbrefID via Chadwick register, and write
    one SeasonStats row per MLB season (batting and pitching merged).

    Returns total rows written.
    """
    import pybaseball as pyb
    from prospects.ingestion.pybaseball_loader import _ensure_lahman

    if not _ensure_lahman():
        if verbose:
            print("[mlb-seasons] Lahman unavailable")
        return 0

    if verbose:
        print("[mlb-seasons] loading Chadwick register + Lahman tables")
    reg = pyb.chadwick_register()
    # STRICT MLBAM matching only (v1.10+). The (last, first) name index
    # was the source of cross-attribution where multiple modern prospects
    # got the MLB stats of a single historical player or vice versa. We
    # now require chadwick.key_mlbam to match our prospect.mlbam_id.
    mlbam_idx: dict[str, "object"] = {}
    for row in reg.itertuples(index=False):
        key_mlbam = getattr(row, "key_mlbam", None)
        if key_mlbam is None:
            continue
        try:
            if isinstance(key_mlbam, float) and np.isnan(key_mlbam):
                continue
            mlbam_idx[str(int(key_mlbam))] = row
        except (TypeError, ValueError):
            continue

    bat = pyb.lahman.batting()
    pit = pyb.lahman.pitching()

    # Aggregate batting per (playerID, yearID) — Lahman gives stints per year
    bat_grp = bat.groupby(["playerID", "yearID"], as_index=False).agg({
        "AB": "sum", "H": "sum", "2B": "sum", "3B": "sum", "HR": "sum",
        "BB": "sum", "HBP": "sum" if "HBP" in bat.columns else "sum",
        "SO": "sum" if "SO" in bat.columns else "sum",
        "SF": "sum" if "SF" in bat.columns else "sum",
        "SB": "sum",
    })
    pit_grp = pit.groupby(["playerID", "yearID"], as_index=False).agg({
        "IPouts": "sum", "ER": "sum", "BB": "sum", "SO": "sum",
        "H": "sum", "HR": "sum",
    })

    # Index by playerID for quick per-player slicing
    bat_by_pid = {pid: g for pid, g in bat_grp.groupby("playerID")}
    pit_by_pid = {pid: g for pid, g in pit_grp.groupby("playerID")}

    csv_fh = None
    csv_writer = None
    if csv_path:
        import csv as _csv, os, time as _time
        from dataclasses import fields, asdict
        new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
        fieldnames = [f.name for f in fields(SeasonStats)] + ["stats_type", "pulled_at"]
        csv_writer = _csv.DictWriter(csv_fh, fieldnames=fieldnames)
        if new_file:
            csv_writer.writeheader()
        if verbose:
            print(f"[mlb-seasons] teeing rows to {csv_path}")

    prospects = db.all_prospects()
    if verbose:
        print(f"[mlb-seasons] scanning {len(prospects)} prospects...")
    total = 0
    matched_players = 0

    suffixes_re = re.compile(r"\b(jr|sr|ii|iii|iv)\b")

    for i, p in enumerate(prospects):
        mlbam = p.get("mlbam_id")
        if not mlbam or mlbam in ("", "-1"):
            continue
        chosen = mlbam_idx.get(str(mlbam))
        if chosen is None:
            continue
        bbref = getattr(chosen, "key_bbref", None)
        if not bbref:
            continue

        b_rows = bat_by_pid.get(bbref)
        p_rows = pit_by_pid.get(bbref)
        if b_rows is None and p_rows is None:
            continue

        matched_players += 1
        years = set()
        if b_rows is not None:
            years.update(b_rows["yearID"].tolist())
        if p_rows is not None:
            years.update(p_rows["yearID"].tolist())

        for year in sorted(years):
            br = b_rows[b_rows["yearID"] == year].iloc[0] if b_rows is not None and (b_rows["yearID"] == year).any() else None
            pr = p_rows[p_rows["yearID"] == year].iloc[0] if p_rows is not None and (p_rows["yearID"] == year).any() else None

            pa = 0
            avg = obp = slg = iso = k_pct = bb_pct = None
            hr = sb = None
            if br is not None:
                AB = float(br.get("AB", 0) or 0)
                H = float(br.get("H", 0) or 0)
                B2 = float(br.get("2B", 0) or 0)
                B3 = float(br.get("3B", 0) or 0)
                HR = float(br.get("HR", 0) or 0)
                BB = float(br.get("BB", 0) or 0)
                HBP = float(br.get("HBP", 0) or 0)
                SO = float(br.get("SO", 0) or 0)
                SF = float(br.get("SF", 0) or 0)
                SBv = float(br.get("SB", 0) or 0)
                pa = int(AB + BB + HBP + SF)
                avg = _safe_div(H, AB)
                obp = _safe_div(H + BB + HBP, pa)
                TB = H + 2 * B2 + 3 * B3 + 4 * HR
                slg = _safe_div(TB, AB)
                if slg is not None and avg is not None:
                    iso = round(slg - avg, 3)
                bb_pct = _safe_div(BB, pa)
                k_pct = _safe_div(SO, pa)
                hr = int(HR)
                sb = int(SBv)

            ip = 0.0; era = whip = k9 = bb9 = hr9 = None
            if pr is not None:
                IPouts = float(pr.get("IPouts", 0) or 0)
                ip = IPouts / 3.0
                ER = float(pr.get("ER", 0) or 0)
                BB_p = float(pr.get("BB", 0) or 0)
                SO_p = float(pr.get("SO", 0) or 0)
                H_p = float(pr.get("H", 0) or 0)
                HR_p = float(pr.get("HR", 0) or 0)
                era = _safe_div(9 * ER, ip)
                whip = _safe_div(BB_p + H_p, ip)
                k9 = _safe_div(9 * SO_p, ip)
                bb9 = _safe_div(9 * BB_p, ip)
                hr9 = _safe_div(9 * HR_p, ip)

            s = SeasonStats(
                player_id=p["player_id"],
                season_year=int(year),
                level="MLB",
                org=None,
                pa=pa, avg=avg, obp=obp, slg=slg, iso=iso,
                k_pct=k_pct, bb_pct=bb_pct,
                home_runs=hr, stolen_bases=sb,
                ip=ip, era=era, whip=whip,
                k9=k9, bb9=bb9, hr9=hr9,
                primary_position=p.get("primary_position"),
            )
            db.upsert_season_stats(s)
            if csv_writer is not None:
                import time as _time
                from dataclasses import asdict
                row = asdict(s)
                row["stats_type"] = "pitching" if pr is not None and br is None else "batting"
                row["pulled_at"] = _time.strftime("%Y-%m-%dT%H:%M:%S")
                csv_writer.writerow(row)
            total += 1

        if verbose and (i + 1) % 2500 == 0:
            print(f"  {i+1}/{len(prospects)} scanned, "
                  f"{matched_players} matched, {total} MLB-season rows")
            if csv_fh:
                csv_fh.flush()

    if csv_fh is not None:
        csv_fh.flush()
        csv_fh.close()
    if verbose:
        print(f"[mlb-seasons] done: matched={matched_players}, rows={total}")
    return total
