"""Encoder upgrade (architecture C, 2026-09-23): transformer and/or ranking tokens.

v3's encoder is a GRU over each player's non-MLB season rows (19 stat columns + presence flags
+ age-of-row). Two independent changes, switchable so a screen can attribute any gain:

  --arch transformer   2-layer TransformerEncoder (d=48, 4 heads, learned positions, causal
                       padding mask), pooled at the last valid season, instead of the GRU
  --rank-tokens        each season row also carries that season's list rankings: best org rank,
                       best overall rank, number of sources ranking him (log-scaled, with
                       missing flags). Point-in-time: a row for season S carries only lists
                       dated S (preseason lists, published before the season), and the encoder
                       only ever reads rows with season <= the snapshot year.

Output contract unchanged: the 6 multi-horizon debut logits, stacked out-of-fold exactly as
exp_v3_inera (fit rows from 3 player-fold encoders, val rows from the full encoder), written as
an embedding cache exp_v3_all reads with --emb-cache.

    python -m prospects.model.train.exp_seq_c --arch transformer --rank-tokens
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import prep_base
from prospects.model.train import exp_seq_d as sq
from prospects.model.train.joint_xgb import _prep_train

_RUN = config.run()
DB = str(config.model_db())
RANK_COLS = ["rk_org", "rk_ovr", "rk_nsrc"]


class TokensRank(sq.Tokens):
    """Season rows + per-season ranking columns appended to the stat block."""

    def __init__(self, db=None, rank_tokens=True):
        super().__init__(db)
        self.n_num = len(sq.NUM)
        if not rank_tokens:
            return
        con = sqlite3.connect(db or DB)
        r = pd.read_sql("SELECT player_id, CAST(substr(as_of,1,4) AS INTEGER) AS yr, source, overall_rank, org_rank "
                        "FROM rankings_history", con)
        con.close()
        g = r.groupby(["player_id", "yr"]).agg(rk_org=("org_rank", "min"), rk_ovr=("overall_rank", "min"),
                                               rk_nsrc=("source", "nunique")).reset_index()
        # rebuild the row table in the SAME order the parent used, then look up each row's season
        con = sqlite3.connect(db or DB)
        rows = pd.read_sql("SELECT player_id, season_year, level FROM season_stats "
                           "WHERE level != 'MLB' AND season_year IS NOT NULL", con)
        con.close()
        rows["lvl"] = rows["level"].map(sq.LEVELS).fillna(0).astype(int)
        rows = rows.sort_values(["player_id", "season_year", "lvl"]).reset_index(drop=True)
        assert len(rows) == len(self.raw), "row table drifted from the parent's"
        m = rows.merge(g, left_on=["player_id", "season_year"], right_on=["player_id", "yr"], how="left")
        extra = np.column_stack([np.log1p(m["rk_org"].to_numpy(float)), np.log1p(m["rk_ovr"].to_numpy(float)),
                                 m["rk_nsrc"].to_numpy(float)]).astype(np.float32)
        self.raw = np.concatenate([self.raw, extra], axis=1)
        self.present = (~np.isnan(self.raw)).astype(np.float32)
        self.n_num = self.raw.shape[1]
        self.mu = np.zeros(self.n_num, np.float32)
        self.sd = np.ones(self.n_num, np.float32)

    def batch(self, pids, snaps):
        n, F = len(pids), self.n_num
        X = np.zeros((n, sq.MAXLEN, 2 * F + 1), np.float32)
        L = np.zeros((n, sq.MAXLEN), np.int64)
        lens = np.ones(n, np.int64)
        for i, (p, S) in enumerate(zip(pids, snaps)):
            sp = self.span.get(p)
            if sp is None:
                continue
            s, e = sp
            e = s + int(np.searchsorted(self.year[s:e], S, side="right"))
            s = max(s, e - sq.MAXLEN)
            k = e - s
            if k <= 0:
                continue
            z = (self.raw[s:e] - self.mu) / self.sd
            X[i, :k, :F] = np.nan_to_num(z)
            X[i, :k, F:2 * F] = self.present[s:e]
            X[i, :k, -1] = (S - self.year[s:e]) / 5.0
            L[i, :k] = self.lvl[s:e]
            lens[i] = k
        return torch.from_numpy(X), torch.from_numpy(L), torch.from_numpy(lens)


class EncoderC(nn.Module):
    def __init__(self, n_in, n_static, arch):
        super().__init__()
        self.arch = arch
        self.lvl = nn.Embedding(7, 4)
        self.inp = nn.Linear(n_in + 4, 48)
        if arch == "transformer":
            self.pos = nn.Embedding(sq.MAXLEN, 48)
            layer = nn.TransformerEncoderLayer(48, 4, dim_feedforward=96, dropout=0.1, batch_first=True)
            self.seq = nn.TransformerEncoder(layer, num_layers=2)
            d_out = 48
        else:
            self.seq = nn.GRU(48, 64, batch_first=True)
            d_out = 64
        self.mix = nn.Sequential(nn.Linear(d_out + n_static, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, sq.LATENT), nn.ReLU())
        self.head = nn.Linear(sq.LATENT, sq.HMAX_ENC)

    def forward(self, X, L, lens, static):
        h = torch.relu(self.inp(torch.cat([X, self.lvl(L)], dim=-1)))
        idx = torch.arange(len(lens))
        if self.arch == "transformer":
            T = h.shape[1]
            h = h + self.pos(torch.arange(T))[None]
            pad = torch.arange(T)[None, :] >= lens[:, None]
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            out = self.seq(h, mask=causal, src_key_padding_mask=pad)
        else:
            out, _ = self.seq(h)
        last = out[idx, lens - 1]
        return self.head(self.mix(torch.cat([last, static], dim=-1)))


def train_c(tok, d, epochs, seed, arch, log, bs=2048):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pl = d.player_id.unique()
    hold = set(rng.choice(pl, size=max(1, len(pl) // 10), replace=False))
    tr, va = d[~d.player_id.isin(hold)], d[d.player_id.isin(hold)]
    net = EncoderC(2 * tok.n_num + 1, len(sq.STATIC), arch)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3 if arch == "transformer" else 2e-3, weight_decay=1e-4)
    ycols = [f"y{h}" for h in range(1, sq.HMAX_ENC + 1)]
    mcols = [f"m{h}" for h in range(1, sq.HMAX_ENC + 1)]

    def run(frame, train):
        net.train(train)
        order = rng.permutation(len(frame)) if train else np.arange(len(frame))
        tot, wsum = 0.0, 0.0
        P, S = frame.player_id.to_numpy(), frame.snap_year.to_numpy()
        Yt = torch.from_numpy(frame[ycols].to_numpy(np.float32))
        Mt = torch.from_numpy(frame[mcols].to_numpy(np.float32))
        St = sq.static_tensor(frame)
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
        ltr, lva = run(tr, True), run(va, False)
        if lva < best - 1e-5:
            best, best_state = lva, {k: v.clone() for k, v in net.state_dict().items()}
        log(f"        epoch {ep + 1}/{epochs}  train {ltr:.4f}  val {lva:.4f}")
    net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def embed_c(net, tok, d, bs=4096):
    P, S = d.player_id.to_numpy(), d.snap_year.to_numpy()
    St = sq.static_tensor(d)
    out = []
    for i in range(0, len(d), bs):
        X, L, lens = tok.batch(P[i:i + bs], S[i:i + bs])
        out.append(net(X, L, lens, St[i:i + bs]).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["gru", "transformer"], default="transformer")
    ap.add_argument("--rank-tokens", action="store_true")
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tag = f"{args.arch}{'_rank' if args.rank_tokens else ''}"
    out = Path(args.out or REPO_ROOT / "runs" / "exp_seq_c" / f"embeddings_{tag}.npz")
    t0 = time.time()
    tick = lambda m: print(f"[C:{tag}] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    fit_base = _prep_train(pd.read_csv(args.fit, low_memory=False), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    val_base = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    snap_cap = int(fit_base.snap_year.max())
    tok = TokensRank(DB, rank_tokens=args.rank_tokens)
    tok.fit_scaler(set(fit_base.player_id), snap_cap)
    pl = np.array(sorted(set(fit_base.player_id)))
    fold_of = {p: i % 3 for i, p in enumerate(np.random.default_rng(11).permutation(pl))}
    fb = fit_base.drop_duplicates(["player_id", "snap_year"]).copy()
    fb["fold"] = fb.player_id.map(fold_of)
    samples = sq.sample_frame(fb, snap_cap)
    samples["fold"] = samples.player_id.map(fold_of)
    tick(f"{len(samples):,} training snapshots; token width {2 * tok.n_num + 1}")
    parts = []
    for k in range(3):
        net = train_c(tok, samples[samples.fold != k], args.epochs, 100 + k, args.arch, lambda m: None)
        b = fb[fb.fold == k]
        parts.append((b[["player_id", "snap_year"]].to_numpy(), embed_c(net, tok, b)))
        tick(f"fold {k}: {len(b):,} fit snapshots embedded out-of-fold")
    net = train_c(tok, samples, args.epochs, 200, args.arch, tick)
    vb = val_base.drop_duplicates(["player_id", "snap_year"])
    parts.append((vb[["player_id", "snap_year"]].to_numpy(), embed_c(net, tok, vb)))
    keys = np.concatenate([p[0] for p in parts])
    emb = np.concatenate([p[1] for p in parts]).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pid=keys[:, 0], snap=keys[:, 1].astype(int), emb=emb)
    tick(f"wrote {out}: {emb.shape}")


if __name__ == "__main__":
    main()
