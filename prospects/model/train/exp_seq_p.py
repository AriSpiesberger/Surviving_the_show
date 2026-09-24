"""Self-supervised pretraining for the sequence encoder (architecture P, 2026-09-23).

The encoder learns from ~4.5k debuts; depth (E2) and augmentation alone (E1) did not help,
which is what a small labelled set predicts. Pretraining uses the stats themselves as the
signal, over every career we may legally read:

  corpus    every player's non-MLB season rows EXCEPT the held-out val players (their stats
            stay unseen, so the in-era comparison with the adopted encoder remains honest);
            that includes recent players whose outcomes have not resolved, whom the supervised
            model never uses
  pretext   (1) masked-season reconstruction: 15% of seasons in each sequence are blanked
            (values and presence flags zeroed) and the trunk reconstructs their standardized
            stats (MSE on present entries); (2) next-season prediction from the last position
  finetune  the pretrained trunk initialises the adopted transformer + ranking-token encoder
            (EncoderE, same shape as C), trained on the debut objective exactly as before
            (optionally with E's augmentation), stacked out-of-fold, written as an embedding
            cache for exp_v3_all --emb-cache.

No outcome ever enters pretraining; each supervised snapshot still reads only seasons <= its
snapshot year.

    python -m prospects.model.train.exp_seq_p --layers 2          # pretrain + finetune
    python -m prospects.model.train.exp_seq_p --layers 4 --noise 0.5 --crop 0.3 --mask 0.3
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
from prospects.model.train.exp_seq_e import EncoderE, TokensAug, embed_e, train_e
from prospects.model.train.joint_xgb import _prep_train

_RUN = config.run()
DB = str(config.model_db())


class Pretrainer(nn.Module):
    """EncoderE's trunk (lvl, inp, pos, seq) + a reconstruction head and a next-season head."""

    def __init__(self, n_in, n_num, layers, d):
        super().__init__()
        self.enc = EncoderE(n_in, len(sq.STATIC), layers=layers, d=d)
        self.recon = nn.Linear(d, n_num)
        self.next = nn.Linear(d, n_num)

    def trunk(self, X, L, lens):
        e = self.enc
        h = torch.relu(e.inp(torch.cat([X, e.lvl(L)], dim=-1)))
        T = h.shape[1]
        h = h + e.pos(torch.arange(T))[None]
        pad = torch.arange(T)[None, :] >= lens[:, None]
        return e.seq(h, src_key_padding_mask=pad)          # bidirectional for the pretext


def pretrain(tok, pids, snaps, epochs, layers, d, seed, log, bs=1024):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    F = tok.n_num
    model = Pretrainer(2 * F + 1, F, layers, d)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(pids)
    for ep in range(epochs):
        order = rng.permutation(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            b = order[i:i + bs]
            X, L, lens = tok.batch(pids[b], snaps[b])
            target, present = X[:, :, :F].clone(), X[:, :, F:2 * F].clone()
            valid = torch.arange(X.shape[1])[None, :] < lens[:, None]
            m = (torch.rand(valid.shape) < 0.15) & valid
            Xm = X.clone()
            Xm[m] = 0.0
            out = model.trunk(Xm, L, lens)
            rec = model.recon(out)
            l_rec = (((rec - target) ** 2) * present * m[..., None]).sum() / (present * m[..., None]).sum().clamp(min=1)
            # next season: predict the last valid season's stats from the position before it
            has2 = lens >= 2
            if has2.any():
                idx = torch.arange(len(lens))[has2]
                prev = out[idx, lens[has2] - 2]
                tgt, pr = target[idx, lens[has2] - 1], present[idx, lens[has2] - 1]
                l_next = (((model.next(prev) - tgt) ** 2) * pr).sum() / pr.sum().clamp(min=1)
            else:
                l_next = torch.zeros(())
            loss = l_rec + l_next
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        log(f"    pretrain epoch {ep + 1}/{epochs}  loss {tot / max(cnt, 1):.4f}")
    return model.enc.state_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dmodel", type=int, default=48)
    ap.add_argument("--pre-epochs", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--crop", type=float, default=0.0)
    ap.add_argument("--mask", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tag = args.tag or f"pre_L{args.layers}_d{args.dmodel}_n{args.noise:g}_c{args.crop:g}_m{args.mask:g}"
    out = REPO_ROOT / "runs" / "exp_seq_p" / f"embeddings_{tag}.npz"
    t0 = time.time()
    tick = lambda m: print(f"[P:{tag}] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    fit_base = _prep_train(pd.read_csv(args.fit, low_memory=False), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    val_base = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    val_players = set(val_base.player_id)
    snap_cap = int(fit_base.snap_year.max())
    tok = TokensAug(DB, args.noise, args.crop, args.mask, seed=5)
    tok.fit_scaler(set(fit_base.player_id), snap_cap)

    # pretraining corpus: every player in the season table except val players, each read as of
    # a random season of his own career (so sequences of all lengths are seen)
    rng = np.random.default_rng(3)
    corpus = [p for p in tok.span if p not in val_players]
    snaps = []
    for p in corpus:
        s, e = tok.span[p]
        snaps.append(int(tok.year[rng.integers(s, e)]))
    pids, snaps = np.array(corpus, dtype=object), np.array(snaps)
    tick(f"pretraining corpus: {len(pids):,} careers (val players excluded); finetune on "
         f"{fit_base.player_id.nunique():,} fit players")
    tok.augment = False
    state = pretrain(tok, pids, snaps, args.pre_epochs, args.layers, args.dmodel, 7, tick)
    tick("pretraining done")

    arch_kw = {"layers": args.layers, "d": args.dmodel, "pool": "last"}
    pl = np.array(sorted(set(fit_base.player_id)))
    fold_of = {p: i % 3 for i, p in enumerate(np.random.default_rng(11).permutation(pl))}
    fb = fit_base.drop_duplicates(["player_id", "snap_year"]).copy()
    fb["fold"] = fb.player_id.map(fold_of)
    samples = sq.sample_frame(fb, snap_cap)
    samples["fold"] = samples.player_id.map(fold_of)

    def finetune(frame, seed, log=lambda m: None):
        # train_e builds a fresh EncoderE; seed it with the pretrained trunk by patching init
        import prospects.model.train.exp_seq_e as E
        orig = E.EncoderE.__init__

        def init(self, *a, **kw):
            orig(self, *a, **kw)
            own = self.state_dict()
            own.update({k: v for k, v in state.items() if k in own and own[k].shape == v.shape
                        and not k.startswith(("head", "mix"))})
            self.load_state_dict(own)
        E.EncoderE.__init__ = init
        try:
            return train_e(tok, frame, args.epochs, seed, arch_kw, log=log)
        finally:
            E.EncoderE.__init__ = orig

    parts = []
    for k in range(3):
        net = finetune(samples[samples.fold != k], 100 + k)
        b = fb[fb.fold == k]
        parts.append((b[["player_id", "snap_year"]].to_numpy(), embed_e(net, tok, b)))
        tick(f"fold {k} fine-tuned + embedded out-of-fold")
    net = finetune(samples, 200, tick)
    vb = val_base.drop_duplicates(["player_id", "snap_year"])
    parts.append((vb[["player_id", "snap_year"]].to_numpy(), embed_e(net, tok, vb)))
    keys = np.concatenate([p[0] for p in parts])
    emb = np.concatenate([p[1] for p in parts]).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pid=keys[:, 0], snap=keys[:, 1].astype(int), emb=emb)
    tick(f"wrote {out}: {emb.shape}")


if __name__ == "__main__":
    main()
