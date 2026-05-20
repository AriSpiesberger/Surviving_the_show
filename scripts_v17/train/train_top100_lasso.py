"""Train Lasso with target = time-decayed TOP_100 only (no MLB_DEBUT).
Train on full fit slice (universe filter doesnt apply because universe excludes
ever-top-100, so target would be all 0). Evaluate on universe val."""
import pandas as pd, sqlite3, pickle, numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

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

fit = prep("v1.17_fit_long.csv")
val = prep("v1.17_val_long.csv")

# target: time-decayed TOP_100 only
def make_top100(df):
    trig = df["trigger_TOP_100_PROSPECT"]
    real = df["realized_TOP_100_PROSPECT"].values
    yrs = (trig - df["snap_year"]).clip(lower=0).fillna(99).values
    return 3.0 * np.clip(3.0 - yrs, 0, None) * real

ytr = make_top100(fit)
print(f"fit: {len(fit):,}  TOP_100 target mean={ytr.mean():.3f} frac>0={(ytr>0).mean():.1%} max={ytr.max():.0f}")

Xtr = fit[FEAT].values; gtr = fit["player_id"].values
sc = StandardScaler().fit(Xtr)
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(Xtr, ytr, gtr))
lasso = LassoCV(cv=splits, alphas=np.logspace(-4, 0, 30), max_iter=20000, n_jobs=-1).fit(sc.transform(Xtr), ytr)
print(f"\nTOP_100-only Lasso alpha: {lasso.alpha_:.5g}")
print("Coefficients:")
for n, c in zip(FEAT, lasso.coef_):
    if abs(c) > 1e-6: print(f"  {n:<42} {c:+.4f}")
print(f"  intercept: {lasso.intercept_:+.4f}")
val["top100_score"] = lasso.predict(sc.transform(val[FEAT].values))
print(f"val score range: [{val.top100_score.min():.3f}, {val.top100_score.max():.3f}]")

with open("models/top100_lasso_v1.17.pkl","wb") as fh:
    pickle.dump({"scaler":sc,"lasso":lasso,"feature_names":FEAT}, fh)
print("saved models/top100_lasso_v1.17.pkl")

# Apply universe filter to val
c = sqlite3.connect("prospects_snapshot.db")
meta = pd.read_sql("SELECT p.player_id, p.draft_round, p.is_international, o.year_top_100 FROM prospects p LEFT JOIN career_outcomes o ON o.player_id=p.player_id", c); c.close()
def bucket(r):
    if int(r.is_international or 0)==1: return "IFA"
    if pd.isna(r.draft_round): return "IFA"
    return "R1" if int(r.draft_round)==1 else "OTHER"
meta["bucket_v"] = meta.apply(bucket, axis=1)
val_uni = val.merge(meta[["player_id","bucket_v","year_top_100"]], on="player_id", how="left")
val_uni = val_uni[(val_uni.bucket_v!="R1") & (val_uni.year_top_100.isna())].copy()

# Load debut-only lasso for comparison
with open("models/debut_lasso_universe_v1.17.pkl","rb") as fh:
    md = pickle.load(fh)
val_uni["debut_score"] = md["lasso"].predict(md["scaler"].transform(val_uni[md["feature_names"]].values))

print(f"\n=== VAL UNIVERSE — top-N realized MLB_DEBUT  (rankers compared) ===")
print(f"{'yip':>3} {'n':>5} {'base':>6}   {'top10 (TOP100)':>15} {'top10 (debut)':>14}   {'top50 (TOP100)':>15} {'top50 (debut)':>14}   {'top100 (TOP100)':>16} {'top100 (debut)':>15}")
print("-"*150)
for yip in range(0, 8):
    sub = val_uni[val_uni.snap_offset==yip]
    if len(sub) < 100: continue
    n=len(sub); base=sub.realized_MLB_DEBUT.mean()
    def topn(col, k):
        kk = min(k, len(sub))
        tp = int(sub.sort_values(col, ascending=False).head(kk).realized_MLB_DEBUT.sum())
        return f"{tp}/{kk} {tp/kk*100:.0f}%"
    print(f"{yip:>3d} {n:>5d} {base*100:>5.1f}%   "
          f"{topn('top100_score',10):>15} {topn('debut_score',10):>14}   "
          f"{topn('top100_score',50):>15} {topn('debut_score',50):>14}   "
          f"{topn('top100_score',100):>16} {topn('debut_score',100):>15}")

# Score-bin x yip (TOP100 ranker)
print(f"\n=== Score-bin × yip (val universe, TOP_100-only lasso) — realized MLB_DEBUT ===")
edges = [-99, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 99]
labels = ["<0","0-0.25","0.25-0.5","0.5-0.75","0.75-1.0","1.0-1.5","1.5-2.0","2.0-3.0","3.0+"]
val_uni["tb"] = pd.cut(val_uni["top100_score"], bins=edges, labels=labels)
rate = val_uni.groupby(["tb","snap_offset"], observed=True)["realized_MLB_DEBUT"].mean().unstack().reindex(labels)
n = val_uni.groupby(["tb","snap_offset"], observed=True)["realized_MLB_DEBUT"].size().unstack().reindex(labels)
keep = [c for c in rate.columns if c <= 7]
print(rate[keep].map(lambda x: f"{x*100:>5.1f}" if pd.notna(x) else "  -  ").to_string())
print(f"\nn per cell:")
print(n[keep].map(lambda x: f"{int(x):>5d}" if pd.notna(x) else "     ").to_string())

# Per-cohort walkforward
print(f"\n=== PER-COHORT walkforward (universe, ranked by TOP_100-only lasso) ===")
df = pd.read_csv("v17_cohorts_2021_2025_long.csv")
df["entry_year"] = df["snap_year"] - df["snap_offset"]
df = df[df["entry_year"].between(2021, 2025)].copy()
df = df.merge(meta[["player_id","bucket_v","year_top_100"]], on="player_id", how="left")
df = df[(df.bucket_v!="R1") & (df.year_top_100.isna())].copy()
c = sqlite3.connect("prospects_snapshot.db")
birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
df = df.merge(birth[["player_id","birth_year"]], on="player_id", how="left")
df["age_at_snap_centered"] = ((df["snap_year"]-df["birth_year"]).fillna(22.0) - 22)
df["years_in_pro"] = df["snap_offset"]
df["yip_centered"] = df["snap_offset"] - 3
for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
    df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
df["top100_score"] = lasso.predict(sc.transform(df[FEAT].values))

print(f"{'cohort':>6} {'snap':>5} {'yip':>4} {'n':>5} {'pos':>4} {'base':>6} {'top10':>11} {'top50':>11} {'top100':>11}")
print("-"*80)
for cohort in [2021, 2022, 2023]:
    for snap_yr in range(cohort, 2027):
        sub = df[(df.entry_year==cohort) & (df.snap_year==snap_yr)].sort_values("top100_score", ascending=False)
        if len(sub) < 50: continue
        yip = snap_yr - cohort
        n=len(sub); pos=int(sub.realized_MLB_DEBUT.sum()); base=pos/n
        def topn(k):
            kk = min(k, len(sub))
            tp = int(sub.head(kk).realized_MLB_DEBUT.sum())
            return f"{tp}/{kk} ({tp/kk*100:.0f}%)"
        print(f"{cohort:>6d} {snap_yr:>5d} {yip:>4d} {n:>5d} {pos:>4d} {base*100:>5.1f}% "
              f"{topn(10):>11} {topn(50):>11} {topn(100):>11}")
    print()
