"""Evaluate the TOP_100-only lasso on its actual target: realized_TOP_100.
Uses full val (NOT universe-filtered, because the universe excludes ever-top-100
by definition — that would make the target all zeros)."""
import pandas as pd, sqlite3, pickle, numpy as np

FEAT = ["p_TOP_100_PROSPECT","p_MLB_DEBUT","p_ESTABLISHED_MLB","p_STAR_PLUS_ELITE",
        "age_at_snap_centered","years_in_pro",
        "p_TOP_100_PROSPECT_x_yip_centered","p_MLB_DEBUT_x_yip_centered",
        "p_ESTABLISHED_MLB_x_yip_centered","p_STAR_PLUS_ELITE_x_yip_centered"]


def prep(long_csv):
    df = pd.read_csv(long_csv)
    df = df[df.entry_year <= 2020].copy()
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT"):
        df = df[df[f"eligible_{ev}"]==1]
    c = sqlite3.connect("prospects_snapshot.db")
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id","birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = ((df["snap_year"]-df["birth_year"]).fillna(22.0) - 22)
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - 3
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
    return df


val = prep("v1.17_val_long.csv")

# Load both lassos
with open("models/top100_lasso_v1.17.pkl","rb") as fh:
    mt = pickle.load(fh)
with open("models/debut_lasso_universe_v1.17.pkl","rb") as fh:
    md = pickle.load(fh)

val["top100_score"] = mt["lasso"].predict(mt["scaler"].transform(val[mt["feature_names"]].values))
val["debut_score"]  = md["lasso"].predict(md["scaler"].transform(val[md["feature_names"]].values))

# Also raw p_TOP_100 for comparison
print(f"val: {len(val):,} rows  realized_TOP_100={int(val.realized_TOP_100_PROSPECT.sum())} ({val.realized_TOP_100_PROSPECT.mean():.1%})")

print(f"\n=== Top-N realized TOP_100 (full val, NOT universe-filtered) ===")
print(f"{'yip':>3} {'n':>5} {'pos':>4} {'base':>6}   "
      f"{'top10 (TOP100)':>15} {'top10 (debut)':>14} {'top10 (p_TOP100)':>17}   "
      f"{'top50 (TOP100)':>15} {'top50 (debut)':>14} {'top50 (p_TOP100)':>17}   "
      f"{'top100 (TOP100)':>16} {'top100 (debut)':>15} {'top100 (p_TOP100)':>18}")
print("-"*180)
for yip in range(0, 8):
    sub = val[val.snap_offset==yip]
    if len(sub) < 100: continue
    n = len(sub); pos = int(sub.realized_TOP_100_PROSPECT.sum()); base = pos/n
    def topn(col, k):
        kk = min(k, len(sub))
        tp = int(sub.sort_values(col, ascending=False).head(kk).realized_TOP_100_PROSPECT.sum())
        return f"{tp}/{kk} {tp/kk*100:.0f}%"
    print(f"{yip:>3d} {n:>5d} {pos:>4d} {base*100:>5.1f}%   "
          f"{topn('top100_score',10):>15} {topn('debut_score',10):>14} {topn('p_TOP_100_PROSPECT',10):>17}   "
          f"{topn('top100_score',50):>15} {topn('debut_score',50):>14} {topn('p_TOP_100_PROSPECT',50):>17}   "
          f"{topn('top100_score',100):>16} {topn('debut_score',100):>15} {topn('p_TOP_100_PROSPECT',100):>18}")

# Score-bin x yip for TOP_100 lasso on FULL val, realized=TOP_100
print(f"\n=== Score-bin × yip (FULL val, TOP_100-only lasso) — realized TOP_100 ===")
edges = [-99, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 99]
labels = ["<0","0-0.25","0.25-0.5","0.5-0.75","0.75-1.0","1.0-1.5","1.5-2.0","2.0-3.0","3.0+"]
val["tb"] = pd.cut(val["top100_score"], bins=edges, labels=labels)
rate = val.groupby(["tb","snap_offset"], observed=True)["realized_TOP_100_PROSPECT"].mean().unstack().reindex(labels)
n = val.groupby(["tb","snap_offset"], observed=True)["realized_TOP_100_PROSPECT"].size().unstack().reindex(labels)
keep = [c for c in rate.columns if c <= 7]
print(rate[keep].map(lambda x: f"{x*100:>5.1f}" if pd.notna(x) else "  -  ").to_string())
print(f"\nn per cell:")
print(n[keep].map(lambda x: f"{int(x):>5d}" if pd.notna(x) else "     ").to_string())

# Per-cohort: full cohort (with R1+top100 since target is top100)
print(f"\n=== PER-COHORT walkforward (FULL, ranked by TOP_100-only lasso) — realized TOP_100 ===")
df = pd.read_csv("v17_cohorts_2021_2025_long.csv")
df["entry_year"] = df["snap_year"] - df["snap_offset"]
df = df[df["entry_year"].between(2021, 2025)].copy()
c = sqlite3.connect("prospects_snapshot.db")
birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
df = df.merge(birth[["player_id","birth_year"]], on="player_id", how="left")
df["age_at_snap_centered"] = ((df["snap_year"]-df["birth_year"]).fillna(22.0) - 22)
df["years_in_pro"] = df["snap_offset"]
df["yip_centered"] = df["snap_offset"] - 3
for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
    df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
df["top100_score"] = mt["lasso"].predict(mt["scaler"].transform(df[FEAT].values))

print(f"{'cohort':>6} {'snap':>5} {'yip':>4} {'n':>5} {'pos':>4} {'base':>6} {'top10':>11} {'top25':>11} {'top50':>11}")
print("-"*80)
for cohort in [2021, 2022, 2023, 2024, 2025]:
    for snap_yr in range(cohort, 2027):
        sub = df[(df.entry_year==cohort) & (df.snap_year==snap_yr)].sort_values("top100_score", ascending=False)
        if len(sub) < 50: continue
        yip = snap_yr - cohort
        n=len(sub); pos=int(sub.realized_TOP_100_PROSPECT.sum()); base=pos/n if n else 0
        def topn(k):
            kk = min(k, len(sub))
            tp = int(sub.head(kk).realized_TOP_100_PROSPECT.sum())
            return f"{tp}/{kk} ({tp/kk*100:.0f}%)"
        print(f"{cohort:>6d} {snap_yr:>5d} {yip:>4d} {n:>5d} {pos:>4d} {base*100:>5.1f}% "
              f"{topn(10):>11} {topn(25):>11} {topn(50):>11}")
    print()
