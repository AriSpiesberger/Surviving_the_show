"""Multi-task sequence encoder: heads for debut, top-100, established and star+ (2026-09-23).

v3's GRU encoder is trained on the debut objective only, so the 6 logits it hands the GBM are
about debut. For the other targets the sheet shows, the encoder should learn from them too:
one GRU trunk, 4 events x 6 horizons = 24 masked-BCE heads (a cell is used only when the
horizon has resolved and the player is eligible for that event at the snapshot). Rare heads
are up-weighted so they are not drowned by debut.

Stacking contract as in v3: training rows get logits from an encoder that never saw that
player (the same 3 player folds as exp_v3_inera), val rows from the full encoder. The output
is an embedding cache that exp_v3_all reads with --emb-cache (its width is inferred).

    python -m prospects.model.train.exp_seq_mt                 # fit + val longs of the current run
    python -m prospects.model.train.exp_seq_mt --sample-frac 0.25
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
from prospects.model.train.exp_seq_d import (HMAX_ENC, LATENT, NUM, STATIC, Encoder, Tokens,
                                             static_tensor)
from prospects.model.train.joint_xgb import _prep_train

_RUN = config.run()
DB = str(config.model_db())
MT_EVENTS = ["MLB_DEBUT", "TOP_100_PROSPECT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
# per-event loss weights: rough inverse sqrt of the positive rate relative to debut
EV_W = {"MLB_DEBUT": 1.0, "TOP_100_PROSPECT": 3.0, "ESTABLISHED_MLB": 1.6, "STAR_PLUS_ELITE": 3.0}


class EncoderMT(Encoder):
    def __init__(self, n_static, n_out):
        super().__init__(n_static)
        self.head = nn.Linear(LATENT, n_out)


def sample_frame_mt(base: pd.DataFrame, snap_cap: int) -> pd.DataFrame:
    d = base[(base.get("eligible_MLB_DEBUT", 1) == 1) & (base.snap_year <= snap_cap)].copy()
    d = d.drop_duplicates(["player_id", "snap_year"])
    for ev in MT_EVENTS:
        trig = pd.to_numeric(d.get(f"trigger_{ev}"), errors="coerce")
        el = (d[f"eligible_{ev}"] == 1) if f"eligible_{ev}" in d.columns else pd.Series(True, index=d.index)
        for h in range(1, HMAX_ENC + 1):
            d[f"y_{ev}_{h}"] = ((trig > d.snap_year) & (trig <= d.snap_year + h)).fillna(False).astype(np.float32)
            d[f"m_{ev}_{h}"] = ((d.years_fwd >= h) & el).astype(np.float32) * EV_W[ev]
    return d


def cols():
    y = [f"y_{ev}_{h}" for ev in MT_EVENTS for h in range(1, HMAX_ENC + 1)]
    m = [f"m_{ev}_{h}" for ev in MT_EVENTS for h in range(1, HMAX_ENC + 1)]
    return y, m


def train_mt(tok, d, epochs, seed, log, bs=2048):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pl = d.player_id.unique()
    hold = set(rng.choice(pl, size=max(1, len(pl) // 10), replace=False))
    tr, va = d[~d.player_id.isin(hold)], d[d.player_id.isin(hold)]
    ycols, mcols = cols()
    net = EncoderMT(len(STATIC), len(ycols))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)

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
        ltr, lva = run(tr, True), run(va, False)
        if lva < best - 1e-5:
            best, best_state = lva, {k: v.clone() for k, v in net.state_dict().items()}
        log(f"        epoch {ep + 1}/{epochs}  train {ltr:.4f}  val {lva:.4f}")
    net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def embed_mt(net, tok, d, bs=4096):
    P, S = d.player_id.to_numpy(), d.snap_year.to_numpy()
    St = static_tensor(d)
    out = []
    for i in range(0, len(d), bs):
        X, L, lens = tok.batch(P[i:i + bs], S[i:i + bs])
        out.append(net(X, L, lens, St[i:i + bs]).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(MT_EVENTS) * HMAX_ENC), np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "exp_seq_mt" / "embeddings_mt.npz"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    tick = lambda m: print(f"[mt] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    fit_base = _prep_train(pd.read_csv(args.fit, low_memory=False), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    val_base = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    snap_cap = int(fit_base.snap_year.max())
    tok = Tokens(DB)
    tok.fit_scaler(set(fit_base.player_id), snap_cap)
    pl = np.array(sorted(set(fit_base.player_id)))
    fold_of = {p: i % 3 for i, p in enumerate(np.random.default_rng(11).permutation(pl))}
    fb = fit_base.drop_duplicates(["player_id", "snap_year"]).copy()
    fb["fold"] = fb.player_id.map(fold_of)
    samples = sample_frame_mt(fb, snap_cap)
    samples["fold"] = samples.player_id.map(fold_of)
    ycols, _ = cols()
    tick(f"{len(samples):,} training snapshots; positives per head (h=6): "
         + ", ".join(f"{ev}={int(samples[f'y_{ev}_6'].sum())}" for ev in MT_EVENTS))
    parts = []
    for k in range(3):
        net = train_mt(tok, samples[samples.fold != k], args.epochs, 100 + k, lambda m: None)
        b = fb[fb.fold == k]
        parts.append((b[["player_id", "snap_year"]].to_numpy(), embed_mt(net, tok, b)))
        tick(f"fold {k}: {len(b):,} fit snapshots embedded out-of-fold")
    net = train_mt(tok, samples, args.epochs, 200, tick)
    vb = val_base.drop_duplicates(["player_id", "snap_year"])
    parts.append((vb[["player_id", "snap_year"]].to_numpy(), embed_mt(net, tok, vb)))
    keys = np.concatenate([p[0] for p in parts])
    emb = np.concatenate([p[1] for p in parts]).astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pid=keys[:, 0], snap=keys[:, 1].astype(int), emb=emb,
                        heads=np.array(ycols))
    tick(f"wrote {args.out}: {emb.shape}")


if __name__ == "__main__":
    main()
