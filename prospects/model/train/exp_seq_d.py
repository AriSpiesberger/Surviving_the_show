"""D: does a learned season-sequence representation beat the hand-built windows? (2026-09-20)

The panel compresses a career into yT / y1 / y2 windows plus career aggregates. D replaces
that human compression with a learned one: a small GRU reads the player's raw non-MLB season
rows as of the snapshot and is trained on the masked multi-horizon debut objective (only
resolved landmark cells). Its penultimate layer is handed to the GBM as extra features.

Judged on the corrected walk-forward (exp_walkforward_h, regime capY): train on everything
known at origin Y, score the entry-(Y, Y+6] cohort at Y+6, label = debut within 3 more years.
The embedding is a second-stage INPUT, so it is stacked the same honest way the hazards are:

  training rows   -> embedding from an encoder trained WITHOUT that player (the same 3 player
                     folds the OOF hazards used: membership = which oof_fold file holds him)
  unseen players  -> embedding from the full-fit encoder (as production would)
  eval snapshot   -> full-fit encoder

Arms (same joint recipe, seeds and eval players as the saved controls):
  A+emb   production stack features + embedding          vs  A_oof_capY
  B1+emb  single stage (no hazard inputs) + embedding    vs  B1
Paired bootstrap on AP per origin and on the mean.

    python -m prospects.model.train.exp_seq_d            # full
    python -m prospects.model.train.exp_seq_d --quick    # smoke test
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from prospects.config import REPO_ROOT
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import H_MAX, prep_base
from prospects.model.joint2 import attach_raw_features
from prospects.model.train.exp_cdf_timing2 import stamp_extra_cols
from prospects.model.train.exp_macro_bc import OUT_DIR as BC_DIR, feature_sets, run_variant
from prospects.model.train.exp_walkforward2 import DB, EVAL_H, GAP
from prospects.model.train.joint_xgb import _assemble

WFH = REPO_ROOT / "runs" / "exp_walkforward_h"
OUT_DIR = REPO_ROOT / "runs" / "exp_seq_d"
LEVELS = {"RK": 1, "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6}
NUM = ["age_during_season", "pa", "ip", "avg", "obp", "slg", "woba", "iso", "k_pct", "bb_pct",
       "babip", "home_runs", "stolen_bases", "era", "fip", "whip", "k9", "bb9", "hr9"]
MAXLEN, HMAX_ENC, LATENT = 16, 6, 32
# What the second stage is fed (2026-09-22). "latent" = the 32-d penultimate layer, as first
# run: its coordinates are private to each trained network, so the 3 fold encoders (training
# rows) and the full encoder (scored rows) spoke unrelated languages — per-dimension means
# correlated at -0.22 in exp_v3_inera, and in-era calibration blew up ~3x. "logits" = the
# encoder's 6 multi-horizon debut logits, whose meaning is the same for every encoder: the
# same stacking contract the hazard layer uses.
SEQ_OUT = "logits"
EMB = HMAX_ENC if SEQ_OUT == "logits" else LATENT


# ---------------------------------------------------------------- tokens
class Tokens:
    """All non-MLB season rows, sorted by (player, year, level); as-of slicing by year."""

    def __init__(self):
        con = sqlite3.connect(DB)
        cols = ", ".join(NUM)
        df = pd.read_sql(f"SELECT player_id, season_year, level, {cols} FROM season_stats "
                         "WHERE level != 'MLB' AND season_year IS NOT NULL", con)
        con.close()
        df["lvl"] = df["level"].map(LEVELS).fillna(0).astype(int)
        df = df.sort_values(["player_id", "season_year", "lvl"]).reset_index(drop=True)
        x = df[NUM].to_numpy(dtype=np.float32)
        x[:, 1] = np.log1p(np.nan_to_num(x[:, 1]))          # pa
        x[:, 2] = np.log1p(np.nan_to_num(x[:, 2]))          # ip
        self.present = (~np.isnan(x)).astype(np.float32)
        self.raw = x
        self.year = df["season_year"].to_numpy(dtype=np.int32)
        self.lvl = df["lvl"].to_numpy(dtype=np.int64)
        pid = df["player_id"].to_numpy()
        starts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1]
        ends = np.r_[starts[1:], len(pid)]
        self.span = {pid[s]: (int(s), int(e)) for s, e in zip(starts, ends)}
        self.mu = np.zeros(len(NUM), np.float32)
        self.sd = np.ones(len(NUM), np.float32)

    def fit_scaler(self, pids, max_year):
        idx = np.concatenate([np.arange(*self.span[p]) for p in pids if p in self.span] or [np.arange(0)])
        idx = idx[self.year[idx] <= max_year]
        self.mu = np.nanmean(self.raw[idx], axis=0).astype(np.float32)
        self.sd = (np.nanstd(self.raw[idx], axis=0) + 1e-6).astype(np.float32)

    def batch(self, pids, snaps):
        n = len(pids)
        X = np.zeros((n, MAXLEN, 2 * len(NUM) + 1), np.float32)
        L = np.zeros((n, MAXLEN), np.int64)
        lens = np.ones(n, np.int64)
        for i, (p, S) in enumerate(zip(pids, snaps)):
            sp = self.span.get(p)
            if sp is None:
                continue
            s, e = sp
            e = s + int(np.searchsorted(self.year[s:e], S, side="right"))   # rows with year <= S
            s = max(s, e - MAXLEN)
            k = e - s
            if k <= 0:
                continue
            z = (self.raw[s:e] - self.mu) / self.sd
            X[i, :k, :len(NUM)] = np.nan_to_num(z)
            X[i, :k, len(NUM):2 * len(NUM)] = self.present[s:e]
            X[i, :k, -1] = (S - self.year[s:e]) / 5.0
            L[i, :k] = self.lvl[s:e]
            lens[i] = k
        return torch.from_numpy(X), torch.from_numpy(L), torch.from_numpy(lens)


class Encoder(nn.Module):
    def __init__(self, n_static):
        super().__init__()
        self.lvl = nn.Embedding(7, 4)
        self.inp = nn.Linear(2 * len(NUM) + 1 + 4, 48)
        self.gru = nn.GRU(48, 64, batch_first=True)
        self.mix = nn.Sequential(nn.Linear(64 + n_static, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, LATENT), nn.ReLU())
        self.head = nn.Linear(LATENT, HMAX_ENC)

    def embed(self, X, L, lens, static):
        h = torch.relu(self.inp(torch.cat([X, self.lvl(L)], dim=-1)))
        out, _ = self.gru(h)
        last = out[torch.arange(len(lens)), lens - 1]
        return self.mix(torch.cat([last, static], dim=-1))

    def forward(self, X, L, lens, static):
        return self.head(self.embed(X, L, lens, static))


# ---------------------------------------------------------------- samples
STATIC = ["age_at_snap_centered", "years_in_pro", "is_ifa", "is_udfa", "draft_round_filled"]


def sample_frame(base: pd.DataFrame, snap_cap: int) -> pd.DataFrame:
    """One row per (player, snapshot) with the multi-horizon debut targets and masks."""
    d = base[(base.get("eligible_MLB_DEBUT", 1) == 1) & (base.snap_year <= snap_cap)].copy()
    d = d.drop_duplicates(["player_id", "snap_year"])
    trig = pd.to_numeric(d["trigger_MLB_DEBUT"], errors="coerce")
    for h in range(1, HMAX_ENC + 1):
        d[f"y{h}"] = ((trig > d.snap_year) & (trig <= d.snap_year + h)).fillna(False).astype(np.float32)
        d[f"m{h}"] = (d.years_fwd >= h).astype(np.float32)
    return d


def static_tensor(d):
    s = d[STATIC].to_numpy(dtype=np.float32).copy()
    s[:, 4] = s[:, 4] / 50.0
    s[:, 1] = s[:, 1] / 5.0
    s[:, 0] = s[:, 0] / 5.0
    return torch.from_numpy(np.nan_to_num(s))


def train_encoder(tok, d, epochs, seed, log, bs=2048):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pl = d.player_id.unique()
    hold = set(rng.choice(pl, size=max(1, len(pl) // 10), replace=False))
    tr, va = d[~d.player_id.isin(hold)], d[d.player_id.isin(hold)]
    net = Encoder(len(STATIC))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    ycols = [f"y{h}" for h in range(1, HMAX_ENC + 1)]
    mcols = [f"m{h}" for h in range(1, HMAX_ENC + 1)]

    def run(frame, train):
        net.train(train)
        order = rng.permutation(len(frame)) if train else np.arange(len(frame))
        tot, wsum = 0.0, 0.0
        P, S = frame.player_id.to_numpy(), frame.snap_year.to_numpy()
        Yt = torch.from_numpy(frame[ycols].to_numpy(np.float32))
        Mt = torch.from_numpy(frame[mcols].to_numpy(np.float32))
        St = static_tensor(frame)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            X, L, lens = tok.batch(P[b], S[b])
            with torch.set_grad_enabled(train):
                logit = net(X, L, lens, St[b])
                loss = (nn.functional.binary_cross_entropy_with_logits(logit, Yt[b], reduction="none")
                        * Mt[b]).sum() / Mt[b].sum().clamp(min=1)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            tot += loss.item() * float(Mt[b].sum())
            wsum += float(Mt[b].sum())
        return tot / max(wsum, 1)

    best, best_state = 1e9, None
    for ep in range(epochs):
        ltr = run(tr, True)
        lva = run(va, False)
        if lva < best - 1e-5:
            best, best_state = lva, {k: v.clone() for k, v in net.state_dict().items()}
        log(f"        epoch {ep + 1}/{epochs}  train {ltr:.4f}  val {lva:.4f}")
    net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def embed(net, tok, d, bs=4096):
    P, S = d.player_id.to_numpy(), d.snap_year.to_numpy()
    St = static_tensor(d)
    out = np.zeros((len(d), EMB), np.float32)
    for i in range(0, len(d), bs):
        X, L, lens = tok.batch(P[i:i + bs], S[i:i + bs])
        o = net(X, L, lens, St[i:i + bs]) if SEQ_OUT == "logits" else net.embed(X, L, lens, St[i:i + bs])
        out[i:i + bs] = o.numpy()
    return out


# ---------------------------------------------------------------- experiment
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origins", nargs="*", type=int, default=[2016, 2014, 2012])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"{m}  [{(time.time() - t0) / 60:.0f}m]", flush=True)

    with open(REPO_ROOT / "runs" / "current" / "models" / "joint_xgb_v2.3.pkl", "rb") as fh:
        keep_raw = list(pickle.load(fh)["keep_raw"])
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    keep_raw = [c for c in keep_raw if c in live]
    fs = feature_sets(keep_raw)
    raw_all = sorted(set(keep_raw) | (set(fs["B"]) & live))
    emb_cols = [f"seq_emb_{i}" for i in range(EMB)]
    tok = Tokens()
    log(f"tokens: {len(tok.year):,} season rows, {len(tok.span):,} players")

    results = []
    for Y in args.origins:
        snap = Y + GAP
        odir = WFH / f"Y{Y}" / "capY"
        log(f"\n===== origin Y={Y} (score snap {snap}) =====")
        folds = [prep_base(pd.read_csv(odir / f"oof_fold{f}.csv", low_memory=False), DB) for f in range(3)]
        rest = prep_base(pd.read_csv(odir / "rest_long.csv", low_memory=False), DB)
        evl = prep_base(pd.read_csv(odir / "eval_long.csv", low_memory=False), DB)
        if args.quick:
            keep = set(np.random.default_rng(0).choice(pd.concat(folds).player_id.unique(), 3000, replace=False))
            folds = [f[f.player_id.isin(keep)] for f in folds]
        train_players = set(pd.concat(folds).player_id.unique())
        tok.fit_scaler(train_players, snap)

        # OOF embeddings for the training players; full-fit encoder for everyone else
        samples = [sample_frame(f, snap - 1) for f in folds]
        emb_parts = []
        for f in range(3):
            d_tr = pd.concat([samples[g] for g in range(3) if g != f], ignore_index=True)
            log(f"    encoder fold {f}: train on {len(d_tr):,} snapshots")
            net = train_encoder(tok, d_tr, 1 if args.quick else args.epochs, 100 + f, log)
            base_f = folds[f].drop_duplicates(["player_id", "snap_year"])
            e = embed(net, tok, base_f)
            emb_parts.append(pd.concat([base_f[["player_id", "snap_year"]].reset_index(drop=True),
                                        pd.DataFrame(e, columns=emb_cols)], axis=1))
        d_all = pd.concat(samples, ignore_index=True)
        log(f"    encoder full: train on {len(d_all):,} snapshots")
        net = train_encoder(tok, d_all, 1 if args.quick else args.epochs, 200, log)
        for frame in (rest, evl):
            b = frame.drop_duplicates(["player_id", "snap_year"])
            b = b[~b.player_id.isin(train_players)] if frame is rest else b
            if len(b):
                e = embed(net, tok, b)
                emb_parts.append(pd.concat([b[["player_id", "snap_year"]].reset_index(drop=True),
                                            pd.DataFrame(e, columns=emb_cols)], axis=1))
        emb_df = pd.concat(emb_parts, ignore_index=True)
        # eval rows must use the full-fit encoder even for players seen in training snapshots
        ev_keys = evl[evl.snap_year == snap][["player_id", "snap_year"]].drop_duplicates()
        e = embed(net, tok, evl[evl.snap_year == snap].drop_duplicates(["player_id", "snap_year"]))
        ev_emb = pd.concat([ev_keys.reset_index(drop=True), pd.DataFrame(e, columns=emb_cols)], axis=1)
        emb_df = pd.concat([emb_df[~((emb_df.snap_year == snap))], ev_emb], ignore_index=True)
        emb_df = emb_df.drop_duplicates(["player_id", "snap_year"], keep="last")
        del net

        both = pd.concat(folds + [rest[rest.snap_year < snap]], ignore_index=True)
        both = both[both.snap_year < snap]
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in both.columns:
                both = both[both[f"eligible_{ev}"] == 1]
        both = attach_raw_features(both, DB, raw_all, verbose=False)
        both = both.merge(emb_df, on=["player_id", "snap_year"], how="left")
        fit_long, Y_fit = _assemble(both, H_MAX)
        fit_long = stamp_extra_cols(fit_long)
        ev_base = evl[(evl.snap_year == snap) & (evl.get("eligible_MLB_DEBUT", 1) == 1)].copy()
        ev_base = attach_raw_features(ev_base, DB, raw_all, verbose=False)
        ev_base = ev_base.merge(emb_df, on=["player_id", "snap_year"], how="left")
        trig = pd.to_numeric(ev_base["trigger_MLB_DEBUT"], errors="coerce")
        y = ((trig > snap) & (trig <= snap + EVAL_H)).fillna(False).to_numpy().astype(int)
        cov = float(fit_long[emb_cols[0]].notna().mean())
        log(f"    joint rows {len(fit_long):,} (embedding coverage {cov:.1%}); eval n={len(y):,} pos={int(y.sum()):,}")

        w = (0.5 ** ((Y - fit_long["snap_year"].to_numpy()) / 4.0)).astype(np.float32)
        arms = {"A+emb": (fs["A"] + emb_cols, None), "B1+emb": (fs["B"] + emb_cols, w)}
        if args.quick:
            arms = {"A+emb": arms["A+emb"]}
        for name, (feats, weight) in arms.items():
            r = run_variant(f"D_{name}", feats, fit_long, Y_fit, ev_base, y, Y, weight, False, t0,
                            lambda m: print(m, flush=True))
            r.update({"Y": Y, "n": int(len(y)), "base": float(y.mean())})
            results.append(r)
            (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
        del fit_long, Y_fit, ev_base, both

    # ---- paired bootstrap against the saved controls --------------------------------
    rng = np.random.default_rng(0)
    pairs = [("D_A+emb", "A_oof_capY"), ("D_B1+emb", "B1")]
    draws = {p: [] for p in pairs}
    print("\n===== D vs control: out-of-era AP, debut <= 3y (paired bootstrap, 400 draws) =====")
    for Y in args.origins:
        for new, old in pairs:
            fa, fb = BC_DIR / f"preds_Y{Y}_{new}.npz", BC_DIR / f"preds_Y{Y}_{old}.npz"
            if not (fa.exists() and fb.exists()):
                continue
            a, b = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
            A = pd.DataFrame({"p": a["cal3"], "y": a["y"]}, index=a["pid"])
            B = pd.DataFrame({"p": b["cal3"]}, index=b["pid"])
            idx = A.index.intersection(B.index)
            yy, pa, pb = A.loc[idx, "y"].to_numpy(), A.loc[idx, "p"].to_numpy(), B.loc[idx, "p"].to_numpy()
            d = []
            for _ in range(400):
                i = rng.integers(0, len(yy), len(yy))
                if yy[i].sum():
                    d.append(average_precision_score(yy[i], pa[i]) - average_precision_score(yy[i], pb[i]))
            d = np.array(d)
            draws[(new, old)].append(d)
            print(f"  Y{Y}  {new:9s} {average_precision_score(yy, pa):.4f}  vs {old:11s} "
                  f"{average_precision_score(yy, pb):.4f}   d={d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, "
                  f"{np.percentile(d, 97.5):+.4f}]  P(>0)={np.mean(d > 0):.2f}")
    for pr, ds in draws.items():
        if len(ds) == len(args.origins) and ds:
            m = np.mean([x[:min(map(len, ds))] for x in ds], axis=0)
            print(f"  MEAN {pr[0]} - {pr[1]}: {m.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, "
                  f"{np.percentile(m, 97.5):+.4f}]  P(>0)={np.mean(m > 0):.2f}")
    log("\ndone")


if __name__ == "__main__":
    main()
