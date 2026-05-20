"""Refit debut lasso + top100 lasso + model B on HONEST 80%-hazard data.

Pipeline:
1. Load v1.17h_fit_long.csv and v1.17h_val_long.csv (scored with 80% hazards).
2. Refit debut lasso (universe filter for training) -> models/debut_lasso_universe_v1.17h.pkl
3. Refit top100 lasso (full fit slice, target=time-decayed TOP_100) -> models/top100_lasso_v1.17h.pkl
4. Refit model B (fit+val combined, GroupKFold OOF) -> models/model_b_outcomes_v1.17h.pkl
5. Validate on val -> per-yip thresholds for >=50% MLB_DEBUT
"""
import pandas as pd, sqlite3, pickle, numpy as np
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from collections import Counter

FEAT = ["p_TOP_100_PROSPECT","p_MLB_DEBUT","p_ESTABLISHED_MLB","p_STAR_PLUS_ELITE",
        "age_at_snap_centered","years_in_pro",
        "p_TOP_100_PROSPECT_x_yip_centered","p_MLB_DEBUT_x_yip_centered",
        "p_ESTABLISHED_MLB_x_yip_centered","p_STAR_PLUS_ELITE_x_yip_centered"]


def add_feats(df, db="prospects_snapshot.db"):
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id","birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = ((df["snap_year"]-df["birth_year"]).fillna(22.0) - 22)
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - 3
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
    return df


def bucket_v(r):
    if int(r.is_international or 0)==1: return "IFA"
    if pd.isna(r.draft_round): return "IFA"
    return "R1" if int(r.draft_round)==1 else "OTHER"


def universe_mask(df):
    c = sqlite3.connect("prospects_snapshot.db")
    meta = pd.read_sql("""SELECT p.player_id, p.draft_round, p.is_international,
                                o.year_top_100 FROM prospects p
                          LEFT JOIN career_outcomes o ON o.player_id = p.player_id""", c); c.close()
    meta["bucket_v"] = meta.apply(bucket_v, axis=1)
    df = df.merge(meta[["player_id","bucket_v","year_top_100"]], on="player_id", how="left")
    return df, (df.bucket_v != "R1") & (df.year_top_100.isna())


print("Loading honest scored data...")
fit = pd.read_csv("v1.17h_fit_long.csv")
val = pd.read_csv("v1.17h_val_long.csv")
print(f"  fit: {len(fit):,}  val: {len(val):,}")

# Add features
fit = add_feats(fit); val = add_feats(val)

# ---- 1. DEBUT lasso (universe-filtered training) ----
print("\n[1] Refitting debut lasso (universe-filtered fit slice)...")
fit_u, mask_f = universe_mask(fit)
fit_u = fit_u[mask_f].copy()
fit_u = fit_u[fit_u["eligible_TOP_100_PROSPECT"]==1]
fit_u = fit_u[fit_u["eligible_MLB_DEBUT"]==1]
fit_u = fit_u[fit_u["entry_year"] <= 2020]

def make_debut_target(df):
    trig = df["trigger_MLB_DEBUT"]
    real = df["realized_MLB_DEBUT"].values
    yrs = (trig - df["snap_year"]).clip(lower=0).fillna(99).values
    return 5.0 * np.clip(4.0 - yrs, 0, None) * real

ytr = make_debut_target(fit_u)
Xtr = fit_u[FEAT].values; gtr = fit_u["player_id"].values
sc = StandardScaler().fit(Xtr)
gkf = GroupKFold(5); splits = list(gkf.split(Xtr, ytr, gtr))
ldb = LassoCV(cv=splits, alphas=np.logspace(-4,0,30), max_iter=20000, n_jobs=-1).fit(sc.transform(Xtr), ytr)
print(f"  debut lasso alpha={ldb.alpha_:.5g}")
for n, c in zip(FEAT, ldb.coef_):
    if abs(c) > 1e-6: print(f"    {n:<42} {c:+.4f}")
with open("models/debut_lasso_universe_v1.17h.pkl","wb") as fh:
    pickle.dump({"scaler":sc,"lasso":ldb,"feature_names":FEAT}, fh)
print(f"  saved models/debut_lasso_universe_v1.17h.pkl")

# ---- 2. TOP_100 lasso (full fit slice, no universe filter) ----
print("\n[2] Refitting top100 lasso (full fit slice)...")
fit_full = fit[(fit.entry_year <= 2020) & (fit.eligible_TOP_100_PROSPECT==1) & (fit.eligible_MLB_DEBUT==1)].copy()

def make_top100_target(df):
    trig = df["trigger_TOP_100_PROSPECT"]
    real = df["realized_TOP_100_PROSPECT"].values
    yrs = (trig - df["snap_year"]).clip(lower=0).fillna(99).values
    return 3.0 * np.clip(3.0 - yrs, 0, None) * real

yt = make_top100_target(fit_full)
Xt = fit_full[FEAT].values; gt = fit_full["player_id"].values
sct = StandardScaler().fit(Xt)
splits2 = list(GroupKFold(5).split(Xt, yt, gt))
ltop = LassoCV(cv=splits2, alphas=np.logspace(-4,0,30), max_iter=20000, n_jobs=-1).fit(sct.transform(Xt), yt)
print(f"  top100 lasso alpha={ltop.alpha_:.5g}")
for n, c in zip(FEAT, ltop.coef_):
    if abs(c) > 1e-6: print(f"    {n:<42} {c:+.4f}")
with open("models/top100_lasso_v1.17h.pkl","wb") as fh:
    pickle.dump({"scaler":sct,"lasso":ltop,"feature_names":FEAT}, fh)
print(f"  saved models/top100_lasso_v1.17h.pkl")

# ---- 3. Model B (fit+val combined, GroupKFold OOF) ----
print("\n[3] Refitting model B (fit+val combined)...")
def load_mlb():
    c = sqlite3.connect("prospects_snapshot.db")
    df = pd.read_sql("SELECT player_id, season_year, pa, ip, woba, era FROM season_stats WHERE UPPER(level)='MLB'", c); c.close()
    df["pa"] = df["pa"].fillna(0); df["ip"] = df["ip"].fillna(0)
    return df
def label_player(rows, debut):
    post = rows[rows.season_year >= debut].copy()
    if post.empty: return "cup"
    if (((post.woba >= 0.350) & (post.pa >= 300)).any()
        or ((post.era <= 3.50) & (post.ip >= 100)).any()): return "breakout"
    if ((post.pa >= 350) | (post.ip >= 80)).any(): return "regular"
    if post.season_year.nunique() >= 2: return "utility"
    career_pa = post.pa.sum()
    came_back = (post.season_year > debut + 1).any()
    if career_pa >= 100 or came_back: return "utility"
    return "cup"
all_long = pd.concat([fit, val], ignore_index=True)
all_long = all_long[all_long.mlb_debut_year.notna()].copy()
all_long["mlb_debut_year"] = all_long["mlb_debut_year"].astype(int)
all_long = all_long[(all_long.mlb_debut_year >= 2010) & (all_long.mlb_debut_year <= 2024)]
all_long = all_long[all_long.snap_year < all_long.mlb_debut_year]
all_long = all_long.sort_values(["player_id","snap_year"]).groupby("player_id").first().reset_index()
mlb = load_mlb()
labels = {pid: label_player(mlb[mlb.player_id==pid], dy) for pid, dy in zip(all_long.player_id, all_long.mlb_debut_year)}
all_long["outcome"] = all_long.player_id.map(labels)
print(f"  debutees: {len(all_long):,}")
for cls, n in Counter(all_long.outcome).most_common(): print(f"    {cls:<10} {n:>4}")

pos_lk = pd.read_csv("models/player_position_from_stats.csv")
all_long = all_long.merge(pos_lk, on="player_id", how="left")
all_long["position"] = all_long["pos_seasonstats"].fillna("UNK")
def pg(p):
    p = str(p).upper()
    if p == "C": return "C"
    if p in ("1B","2B","3B","SS"): return "IF"
    if p in ("LF","CF","RF","OF","DH"): return "OF"
    return "OTH"
all_long["pos_grp"] = all_long["position"].apply(pg)

haz_cols = ["p_TOP_100_PROSPECT","p_MLB_DEBUT","p_ESTABLISHED_MLB","p_STAR_PLUS_ELITE"]
eps = 1e-6
for c in haz_cols:
    all_long[f"logit_{c}"] = np.log(all_long[c].clip(eps,1-eps)/(1-all_long[c].clip(eps,1-eps)))
bdum = pd.get_dummies(all_long["bucket"], prefix="b", drop_first=True)
pdum = pd.get_dummies(all_long["pos_grp"], prefix="pos", drop_first=True)
feat_b = pd.concat([
    all_long[[f"logit_{c}" for c in haz_cols]].reset_index(drop=True),
    all_long[["snap_offset"]].reset_index(drop=True).rename(columns={"snap_offset":"yrs_pre_debut"}),
    bdum.reset_index(drop=True), pdum.reset_index(drop=True),
], axis=1).astype(float)
Xb = feat_b.values; yb_str = all_long["outcome"].values
classes = ["cup","utility","regular","breakout"]
yb = np.array([classes.index(c) for c in yb_str])
gb = all_long.player_id.values
gkf_b = GroupKFold(5)
oof = np.zeros((len(yb), len(classes)))
for tr, te in gkf_b.split(Xb, yb, gb):
    scb = StandardScaler().fit(Xb[tr])
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000).fit(scb.transform(Xb[tr]), yb[tr])
    oof[te] = clf.predict_proba(scb.transform(Xb[te]))
oof_ll = -np.log(oof[np.arange(len(yb)), yb] + 1e-9).mean()
base_rates = np.array([np.mean(yb == k) for k in range(len(classes))])
base_ll = -np.log(base_rates[yb] + 1e-9).mean()
print(f"  OOF log-loss: {oof_ll:.4f}  baseline: {base_ll:.4f}  improvement: {base_ll-oof_ll:.4f}")
sc_full = StandardScaler().fit(Xb)
clf_full = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000).fit(sc_full.transform(Xb), yb)
with open("models/model_b_outcomes_v1.17h.pkl","wb") as fh:
    pickle.dump({"model":clf_full,"scaler":sc_full,"feature_names":list(feat_b.columns),
                 "classes":classes,"oof_log_loss":oof_ll,"baseline_log_loss":base_ll}, fh)
print("  saved models/model_b_outcomes_v1.17h.pkl")

# ---- 4. Honest val + per-yip thresholds ----
print("\n[4] Honest validation: per-yip thresholds for >=50% MLB_DEBUT")
val_u, mask_v = universe_mask(val)
val_u = val_u[mask_v].copy()
val_u = val_u[val_u.entry_year <= 2020]
for ev in ("TOP_100_PROSPECT","MLB_DEBUT"):
    val_u = val_u[val_u[f"eligible_{ev}"]==1]
val_u["lasso_score"] = ldb.predict(sc.transform(val_u[FEAT].values))
print(f"  val universe: {len(val_u):,} / {val_u.player_id.nunique():,} players")
print(f"  score range: [{val_u.lasso_score.min():.3f}, {val_u.lasso_score.max():.3f}]")

print(f"\n  Per-yip cumulative-from-top threshold (lowest score s.t. cum >=50% with n>=20):")
print(f"  {'yip':>4} {'n_total':>7} {'base':>6} | {'score_thresh':>13} {'players above':>14} {'realized %':>11}")
thresholds = {}
for yip in range(0, 7):
    sub = val_u[val_u.snap_offset == yip].sort_values("lasso_score", ascending=False).reset_index(drop=True)
    if len(sub) < 50: continue
    n_total = len(sub); base = sub.realized_MLB_DEBUT.mean()
    cum = sub["realized_MLB_DEBUT"].cumsum().values
    counts = np.arange(1, len(sub)+1)
    rates = cum / counts
    valid = (rates >= 0.5) & (counts >= 20)
    if valid.any():
        k = counts[valid].max()
        thresh = sub["lasso_score"].iloc[k-1]
        tp = int(cum[k-1]); r = rates[k-1]
        thresholds[yip] = float(thresh)
        print(f"  {yip:>4d} {n_total:>7d} {base*100:>5.1f}% | {thresh:>+13.3f} {tp}/{k:>13} {r*100:>10.1f}%")
    else:
        print(f"  {yip:>4d} {n_total:>7d} {base*100:>5.1f}% | no threshold reaches >=50% with n>=20")

print(f"\nFINAL per-yip thresholds (use these in buy list):")
for yip, t in thresholds.items():
    print(f"  yip {yip}: lasso_score >= {t:.3f}")

import json
with open("v17h_thresholds.json","w") as fh:
    json.dump({str(k): v for k,v in thresholds.items()}, fh, indent=2)
print(f"\nsaved v17h_thresholds.json")
