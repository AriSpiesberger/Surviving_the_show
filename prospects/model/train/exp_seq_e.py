"""Experiment E: data augmentation + transformer variants for the sequence encoder (2026-09-23).

The user's idea: a season line is one draw of a random process, so it can be fuzzed; seasons
can be cropped or dropped (missingness is cheap because every token carries presence flags);
careers can be resampled. Built on architecture C (transformer + ranking tokens, adopted
2026-09-23). Augmentation is applied ONLY to training batches; out-of-fold and val embeddings are
always computed on clean data, so the stacking contract is unchanged.

  --noise P     with prob P per training sample, resample every season's stats from their
                sampling distribution at the observed volume: proportions (AVG/OBP/K%/BB%/BABIP)
                ~ N(p, p(1-p)/PA); SLG/ISO/wOBA with per-PA variance 0.5/0.3/0.25; HR/SB
                Poisson-like; ERA/FIP/K9/BB9/HR9 ~ N(r, 9r/IP); WHIP ~ N(w, w/IP)
  --crop P      with prob P, cut the latest season to u ~ U(0.3, 1) of its PA/IP (counts scale
                with it) before the noise, which then follows the smaller sample
  --mask P      with prob P, drop each earlier season with prob 0.15 and blank one stat block
                (hitting or pitching) with prob 0.1
  --layers/--dmodel/--pool   transformer depth, width, pooling (last token | mean over seasons)

Writes an embedding cache (6 debut logits) for exp_v3_all --emb-cache.

    python -m prospects.model.train.exp_seq_e --noise 0.5 --crop 0.3 --mask 0.3
"""
from __future__ import annotations

import argparse
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
from prospects.model.train.exp_seq_c import TokensRank
from prospects.model.train.joint_xgb import _prep_train

_RUN = config.run()
DB = str(config.model_db())
PROP = [3, 4, 8, 9, 10]                       # avg obp k_pct bb_pct babip
PER_PA_VAR = {5: 0.5, 7: 0.3, 6: 0.25}        # slg iso woba
COUNTS = [11, 12]                             # home_runs stolen_bases
PER9 = [13, 14, 16, 17, 18]                   # era fip k9 bb9 hr9
WHIP = 15
HIT_BLOCK, PIT_BLOCK = list(range(3, 13)), list(range(13, 19))


class TokensAug(TokensRank):
    def __init__(self, db, noise=0.0, crop=0.0, mask=0.0, seed=0):
        super().__init__(db, rank_tokens=True)
        self.p_noise, self.p_crop, self.p_mask = noise, crop, mask
        self.augment = False
        self.rng = np.random.default_rng(seed)

    def _perturb(self, x):
        """x: (k, F) raw rows (pa, ip already log1p). Returns a perturbed copy."""
        r = self.rng
        x = x.copy()
        k = len(x)
        if self.p_crop and r.random() < self.p_crop:
            u = r.uniform(0.3, 1.0)
            last = x[-1]
            for c in (1, 2):
                if np.isfinite(last[c]):
                    last[c] = np.log1p(np.expm1(last[c]) * u)
            for c in COUNTS:
                if np.isfinite(last[c]):
                    last[c] *= u
            do_noise = True
        else:
            do_noise = bool(self.p_noise) and r.random() < self.p_noise
        if do_noise:
            pa = np.maximum(np.expm1(np.nan_to_num(x[:, 1])), 1.0)
            ip = np.maximum(np.expm1(np.nan_to_num(x[:, 2])), 1.0)
            for c in PROP:
                p = np.clip(x[:, c], 0.001, 0.999)
                x[:, c] = x[:, c] + r.normal(0, 1, k) * np.sqrt(p * (1 - p) / pa)
            for c, v in PER_PA_VAR.items():
                x[:, c] = x[:, c] + r.normal(0, 1, k) * np.sqrt(v / pa)
            for c in COUNTS:
                x[:, c] = np.maximum(x[:, c] + r.normal(0, 1, k) * np.sqrt(np.maximum(x[:, c], 1.0)), 0)
            for c in PER9:
                x[:, c] = np.maximum(x[:, c] + r.normal(0, 1, k) * np.sqrt(9 * np.maximum(x[:, c], 0.1) / ip), 0)
            x[:, WHIP] = np.maximum(x[:, WHIP] + r.normal(0, 1, k) * np.sqrt(np.maximum(x[:, WHIP], 0.3) / ip), 0)
        if self.p_mask and r.random() < self.p_mask:
            if k > 1:
                drop = r.random(k - 1) < 0.15
                x[:-1][drop] = np.nan
            if r.random() < 0.1:
                x[:, HIT_BLOCK if r.random() < 0.5 else PIT_BLOCK] = np.nan
        return x

    def batch(self, pids, snaps):
        if not self.augment:
            return super().batch(pids, snaps)
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
            raw = self._perturb(self.raw[s:e])
            X[i, :k, :F] = np.nan_to_num((raw - self.mu) / self.sd)
            X[i, :k, F:2 * F] = (~np.isnan(raw)).astype(np.float32)
            X[i, :k, -1] = (S - self.year[s:e]) / 5.0
            L[i, :k] = self.lvl[s:e]
            lens[i] = k
        return torch.from_numpy(X), torch.from_numpy(L), torch.from_numpy(lens)


class EncoderE(nn.Module):
    def __init__(self, n_in, n_static, layers=2, d=48, pool="last"):
        super().__init__()
        self.pool = pool
        self.lvl = nn.Embedding(7, 4)
        self.inp = nn.Linear(n_in + 4, d)
        self.pos = nn.Embedding(sq.MAXLEN, d)
        layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
        self.seq = nn.TransformerEncoder(layer, num_layers=layers)
        self.mix = nn.Sequential(nn.Linear(d + n_static, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, sq.LATENT), nn.ReLU())
        self.head = nn.Linear(sq.LATENT, sq.HMAX_ENC)

    def forward(self, X, L, lens, static):
        h = torch.relu(self.inp(torch.cat([X, self.lvl(L)], dim=-1)))
        T = h.shape[1]
        h = h + self.pos(torch.arange(T))[None]
        pad = torch.arange(T)[None, :] >= lens[:, None]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        out = self.seq(h, mask=causal, src_key_padding_mask=pad)
        if self.pool == "mean":
            keep = (~pad).float()[..., None]
            z = (out * keep).sum(1) / keep.sum(1).clamp(min=1)
        else:
            z = out[torch.arange(len(lens)), lens - 1]
        return self.head(self.mix(torch.cat([z, static], dim=-1)))


def train_e(tok, d, epochs, seed, arch_kw, bs=2048, log=lambda m: None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pl = d.player_id.unique()
    hold = set(rng.choice(pl, size=max(1, len(pl) // 10), replace=False))
    tr, va = d[~d.player_id.isin(hold)], d[d.player_id.isin(hold)]
    net = EncoderE(2 * tok.n_num + 1, len(sq.STATIC), **arch_kw)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ycols = [f"y{h}" for h in range(1, sq.HMAX_ENC + 1)]
    mcols = [f"m{h}" for h in range(1, sq.HMAX_ENC + 1)]

    def run(frame, train):
        net.train(train)
        tok.augment = train                     # augmentation on training batches only
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
        tok.augment = False
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
def embed_e(net, tok, d, bs=4096):
    tok.augment = False
    P, S = d.player_id.to_numpy(), d.snap_year.to_numpy()
    St = sq.static_tensor(d)
    out = [net(*tok.batch(P[i:i + bs], S[i:i + bs]), St[i:i + bs]).numpy() for i in range(0, len(d), bs)]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--crop", type=float, default=0.0)
    ap.add_argument("--mask", type=float, default=0.0)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dmodel", type=int, default=48)
    ap.add_argument("--pool", choices=["last", "mean"], default="last")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tag = args.tag or (f"n{args.noise:g}_c{args.crop:g}_m{args.mask:g}_L{args.layers}_d{args.dmodel}_{args.pool}")
    out = REPO_ROOT / "runs" / "exp_seq_e" / f"embeddings_{tag}.npz"
    t0 = time.time()
    tick = lambda m: print(f"[E:{tag}] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)
    arch_kw = {"layers": args.layers, "d": args.dmodel, "pool": args.pool}

    fit_base = _prep_train(pd.read_csv(args.fit, low_memory=False), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    val_base = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    snap_cap = int(fit_base.snap_year.max())
    tok = TokensAug(DB, args.noise, args.crop, args.mask, seed=5)
    tok.fit_scaler(set(fit_base.player_id), snap_cap)
    pl = np.array(sorted(set(fit_base.player_id)))
    fold_of = {p: i % 3 for i, p in enumerate(np.random.default_rng(11).permutation(pl))}
    fb = fit_base.drop_duplicates(["player_id", "snap_year"]).copy()
    fb["fold"] = fb.player_id.map(fold_of)
    samples = sq.sample_frame(fb, snap_cap)
    samples["fold"] = samples.player_id.map(fold_of)
    tick(f"{len(samples):,} training snapshots")
    parts = []
    for k in range(3):
        net = train_e(tok, samples[samples.fold != k], args.epochs, 100 + k, arch_kw)
        b = fb[fb.fold == k]
        parts.append((b[["player_id", "snap_year"]].to_numpy(), embed_e(net, tok, b)))
        tick(f"fold {k} embedded out-of-fold")
    net = train_e(tok, samples, args.epochs, 200, arch_kw, log=tick)
    vb = val_base.drop_duplicates(["player_id", "snap_year"])
    parts.append((vb[["player_id", "snap_year"]].to_numpy(), embed_e(net, tok, vb)))
    keys = np.concatenate([p[0] for p in parts])
    emb = np.concatenate([p[1] for p in parts]).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pid=keys[:, 0], snap=keys[:, 1].astype(int), emb=emb)
    tick(f"wrote {out}: {emb.shape}")


if __name__ == "__main__":
    main()
