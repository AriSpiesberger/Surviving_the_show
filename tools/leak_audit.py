"""Leak audit -- run after ANY data refresh or feature change, before trusting a retrain.

Validation cannot see a leak that is present in the held-out rows too and absent
at serve time (scout_servicetime, found 2026-09-17, looked like excellent
calibration). These checks therefore never consult the model's metrics:

  1 future-blindness   features are identical with everything after the snapshot
                       year deleted (later seasons, later rankings, outcome and
                       current-state fields)
  2 outcome coverage   no feature is POPULATED more often for eventual debuters
                       than never-debuters at the same career stage, beyond what
                       level / ranking / scouting status legitimately explains
  3 source stamping    no scouting column is one value copied onto every season
                       (a live join from today), and nothing MLB-side is a feature
  4 identity           no real player sits on both sides of the fit/val split
  5 deployed bundles   no banned feature is inside a promoted model

    python tools/leak_audit.py            # exit 1 on any FAIL
"""
from __future__ import annotations

import copy
import pickle
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prospects import config                                          # noqa: E402
from prospects.features import grades                                 # noqa: E402
from prospects.features.scouting import FEATURE_NAMES                 # noqa: E402
from prospects.model.hazards import landmark as lm                    # noqa: E402
from prospects.model.train.score_recent_cohorts import load_universe  # noqa: E402

DB = str(config.model_db())
RUN = config.run()

OUTCOME_KEYS = ["mlb_debut_year", "year_established_mlb", "year_top_100",
                "year_top_25", "year_all_star_once", "year_all_star_three",
                "year_major_award", "year_hof_trajectory", "events_json",
                "final_mlb_year"]
STATE_KEYS = ["current_org", "current_level", "highest_level_reached",
              "has_current_injury", "current_injury_type", "tj_history",
              "notes", "as_of_date", "updated_at"]
# Presence differences these explain are information known at the time, not
# sourcing. *_vs_level: there is no RK / A- league baseline for anyone.
LEGIT = ("org_rank", "top100", "scout_", "reached_", "years_to_",
         "age_at_first", "max_level", "pct_pa_at", "pct_ip_at", "repeat_level",
         "accel_", "delta_", "_y1", "_y2", "level_change", "promotion",
         "best_", "current_", "since_max", "bottom_since", "pa_at_", "ip_at_",
         "_vs_level")
# Pedigree fields gated to a completely-covered block (features/pedigree_rules)
# are a deterministic function of draft round and year.
GATED = ("log_signing_bonus", "bonus_vs_slot", "log_pick_value")
PITCH_STAT = ("ip", "era", "fip", "whip", "k9", "bb9", "hr9", "k_bb")
# Scouting columns that are legitimately fixed for a player.
STATIC_OK = {"draft_rnd", "amateur_rk", "class_rk", "signed_yr", "signed_mkt",
             "school_type", "sign_bonus", "bats", "throws", "height", "had_tj",
             "fypd_eligible"}
PITCHER_POS = {"P", "RHP", "LHP", "SP", "RP"}

fails: list[str] = []


def report(ok: bool, name: str, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append(name)


prospects, stats = load_universe(DB)
names = list(lm.FEATURE_NAMES_LM)[:lm.N_FEATURES]


def feat(p, st, S):
    return np.asarray(lm.build_windowed_features(p, st, S, milb_only=True),
                      dtype=float)


def is_pitcher(p) -> bool:
    return ((p.get("primary_position") or "").upper() in PITCHER_POS
            or bool(p.get("is_pitcher")))


# ---- 1. future-blindness ----------------------------------------------------
random.seed(7)
cand = []
for p in prospects:
    d = p.get("mlb_debut_year")
    if d and 2008 <= d <= 2024:
        yrs = {s["season_year"] for s in stats.get(p["player_id"], [])
               if s.get("season_year")}
        cand += [(p, S) for S in yrs if 2007 <= S < d and S <= 2022]
random.shuffle(cand)
cand = cand[:800]
bad = np.zeros(len(names), int)
for p, S in cand:
    st = stats.get(p["player_id"], [])
    q = copy.deepcopy(p)
    for k in OUTCOME_KEYS + STATE_KEYS:
        if k in q:
            q[k] = None
    for k in ("_top100_rankings", "_org_rankings"):
        if q.get(k):
            q[k] = [r for r in q[k] if int(str(r[0])[:4]) <= S]
    a = feat(p, st, S)
    b = feat(q, [s for s in st if (s.get("season_year") or 0) <= S], S)
    bad += ~((a == b) | (np.isnan(a) & np.isnan(b)))
changed = [names[j] for j in np.where(bad > 0)[0]]
report(not changed, "future-blindness",
       f"{len(cand)} pre-debut snapshots; features that change when the "
       f"future is deleted: {changed[:6] or 'none'}")

# ---- 2. outcome-dependent coverage -----------------------------------------
worst = []
for want_pit in (False, True):
    random.seed(3)
    rows = []
    for p in prospects:
        dy, rd = p.get("draft_year"), p.get("draft_round") or 99
        if (not dy or not (2006 <= dy <= 2020) or rd < 4
                or int(p.get("is_international") or 0)
                or is_pitcher(p) != want_pit):
            continue
        S, d = int(dy) + 1, p.get("mlb_debut_year")
        if d and d <= S:
            continue
        if any(s.get("season_year") == S and (s.get("level") or "") != "MLB"
               for s in stats.get(p["player_id"], [])):
            rows.append((p, S, bool(d)))
    random.shuffle(rows)
    D = [r for r in rows if r[2]][:700]
    N = [r for r in rows if not r[2]][:1400]
    XD = np.vstack([feat(p, stats.get(p["player_id"], []), S) for p, S, _ in D])
    XN = np.vstack([feat(p, stats.get(p["player_id"], []), S) for p, S, _ in N])
    gap = (~np.isnan(XD)).mean(0) - (~np.isnan(XN)).mean(0)
    for j, n in enumerate(names):
        if abs(gap[j]) < 0.08 or n in GATED or any(k in n for k in LEGIT):
            continue
        # the other player type's stat block: two-way reps / position players
        # pitching track level reached, not sourcing
        if any(k in n for k in PITCH_STAT) != want_pit:
            continue
        worst.append((n, round(float(gap[j]), 3), "pit" if want_pit else "hit"))
report(not worst, "outcome-dependent coverage",
       f"unexplained presence gaps >= .08 at draft+1: {worst[:8] or 'none'}")

block = [p for p in prospects
         if (p.get("draft_year") or 0) >= 2017
         and (p.get("draft_round") or 99) <= 10
         and not int(p.get("is_international") or 0)
         and any((s.get("level") or "") != "MLB"
                 for s in stats.get(p["player_id"], []))]


def coverage(ps):
    return (float(np.mean([(p.get("signing_bonus_usd") or 0) > 0 for p in ps]))
            if ps else float("nan"))


cd = coverage([p for p in block if p.get("mlb_debut_year")])
cn = coverage([p for p in block if not p.get("mlb_debut_year")])
report(min(cd, cn) >= 0.995, "bonus block completeness",
       f"2017+ R1-10 coverage: debuted {cd:.3f}, never {cn:.3f}")

# ---- 3. source stamping -----------------------------------------------------
sc = pd.read_csv(config.SCOUTING_GRADES, low_memory=False)
stamped = []
for col in sc.columns:
    if col in ("player_id", "season", "source") or col in STATIC_OK:
        continue
    if not pd.api.types.is_numeric_dtype(sc[col]):
        continue
    g = sc[sc[col].notna()].groupby("player_id")[col].agg(nu="nunique", n="size")
    g = g[g.n >= 3]
    if len(g) >= 100 and (g.nu == 1).mean() >= 0.97:
        stamped.append(col)
live = [c for c in stamped if f"scout_{c}" in FEATURE_NAMES]
report(not live, "scouting stamping",
       f"one-value-per-player columns still used as features: {live or 'none'}"
       f" (detected and banned: {[c for c in stamped if c not in live]})")
# career_milb_* / career_max_* are cumulative MINOR-league totals as of the
# snapshot (covered by the future-blindness check); anything else career- or
# MLB-flavoured would be outcome data.
mlb_side = [n for n in FEATURE_NAMES
            if any(k in n.lower() for k in ("service", "debut", "mlb_", "career_"))
            and not n.lower().startswith(("career_milb_", "career_max_"))]
report(not mlb_side, "MLB-side feature names", f"{mlb_side or 'none'}")

# ---- 3b. impossible ages (welded namesake careers) ------------------------
_c = sqlite3.connect(DB)
young = _c.execute("SELECT COUNT(*) FROM season_stats WHERE age_during_season < 15").fetchone()[0]
early = _c.execute(
    "SELECT COUNT(*) FROM career_outcomes o JOIN prospects p USING(player_id) "
    "WHERE o.mlb_debut_year IS NOT NULL AND p.birth_date IS NOT NULL "
    "AND o.mlb_debut_year < CAST(substr(p.birth_date,1,4) AS INTEGER) + 17").fetchone()[0]
from prospects.data.backfills.repair_impossible_rankings import _BAD_RH, MIN_RANK_AGE
bad_rank = len(_c.execute(_BAD_RH, {"age": MIN_RANK_AGE}).fetchall())
_c.close()
report(young == 0 and early == 0, "impossible ages",
       f"season rows before age 15: {young}; MLB debuts before age 17: {early} "
       f"(fix: data/backfills/repair_impossible_ages)")
report(bad_rank == 0, "impossible rankings",
       f"list rankings before age {MIN_RANK_AGE} or on/before the draft year: {bad_rank} "
       f"(fix: data/backfills/repair_impossible_rankings)")

# ---- 4. identity across the split ------------------------------------------
con = sqlite3.connect(DB)
P = pd.read_sql("select player_id, name, birth_date, mlbam_id from prospects", con)
fit = set(RUN.fit_pids.read_text().split())
val = set(RUN.val_pids.read_text().split())
P["k1"] = P.mlbam_id.astype(str).str.replace(r"\.0$", "", regex=True)
P["k2"] = P.name.str.lower().str.strip() + "|" + P.birth_date.astype(str).str[:10]
straddle = 0
for k in ("k1", "k2"):
    sub = P[P[k].notna() & ~P[k].isin(["None", "nan"]) & P.duplicated(k, keep=False)]
    for _, g in sub.groupby(k):
        ids = set(g.player_id)
        straddle += bool(ids & fit) and bool(ids & val)
report(len(fit & val) == 0 and straddle == 0, "identity across fit/val",
       f"id overlap {len(fit & val)}; same-person groups straddling {straddle}")

# ---- 5. deployed bundles ----------------------------------------------------
banned = {f"scout_{c}" for c in getattr(grades, "_BANNED_COLS", set())}
for f in sorted(RUN.models.glob("joint_xgb_v2.[45].pkl")):
    with open(f, "rb") as fh:
        bundle = pickle.load(fh)
    hit = [n for n in bundle.get("keep_raw", []) if n.replace("rw_", "", 1) in banned]
    report(not hit, f"bundle {f.name}", f"banned features inside: {hit or 'none'}")
hz = RUN.models / "hazards.pkl"
if hz.exists():
    with open(hz, "rb") as fh:
        h = pickle.load(fh)
    fn = next(v["feature_names"] for v in h.values()
              if isinstance(v, dict) and "feature_names" in v)
    hit = [n for n in fn if n in banned]
    report(not hit, "bundle hazards.pkl", f"banned features inside: {hit or 'none'}")
    # The production hazards score the live sheet. They must match the current
    # feature contract and be no older than the panel they claim to come from
    # (train.hazards skips silently when the file exists unless --force).
    same = list(fn) == list(lm.FEATURE_NAMES_LM)
    report(same, "hazards.pkl feature contract",
           f"{len(fn)} features in the model vs {len(lm.FEATURE_NAMES_LM)} in the builder")
    panel = RUN.scratch / "oof" / "panel_cache.npz"
    if panel.exists():
        fresh = hz.stat().st_mtime >= panel.stat().st_mtime
        report(fresh, "hazards.pkl freshness",
               "trained after the current panel" if fresh
               else "OLDER than the panel cache: the live sheet is scored by a stale model")

print("\n" + ("ALL CHECKS PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
