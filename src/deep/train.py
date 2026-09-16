"""Deep models under subject-wise CV (same folds as every model). Validation subjects choose the checkpoint
(best macro-F1); nothing is tuned on the test subject. Fold k uses seed + k with deterministic cuDNN, so a
rerun is bit-identical. Checkpoints: checkpoints/<experiment>/<model>/seed<seed>/fold<k>.pt (git-ignored).

CLI:  uv run python -m src.deep.train --experiment exp1_main --model tinysleepnet --seed 42   # GPU, ~1-2 h
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tF

from src import splits as S
from src.config import load_config, repo_path
from src.evaluate import N_CLASSES, metrics_from_confusion, subject_confusions
from src.runs import predictions_frame, run_dir, save_run, set_seeds


def prepare(ds: dict[str, np.ndarray], subjects: Sequence[int], mode: str) -> dict[str, np.ndarray]:
    """Rows of ``subjects``, scaled per recording: robust median / IQR of its whole night (no labels), or raw µV."""
    m = np.isin(ds["subject"], subjects)
    out = {k: v[m] for k, v in ds.items()}
    if mode == "per_recording_full":
        out["X"] = (out["X"] - out["rec_center"][:, :, None]) / out["rec_scale"][:, :, None]
    out["X"] = np.ascontiguousarray(out["X"], dtype=np.float32)
    assert mode in ("per_recording_full", "none") and np.isfinite(out["X"]).all()
    return out


def recording_slices(rec_id: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], np.flatnonzero(np.diff(rec_id)) + 1])
    assert len(np.unique(rec_id)) == len(starts), "recordings are not contiguous blocks"
    return list(zip(starts.tolist(), np.concatenate([starts[1:], [len(rec_id)]]).tolist()))


def window_starts(slices: list[tuple[int, int]], length: int, stride: int, offset: int = 0) -> np.ndarray:
    """Starts of L-epoch windows that never cross a recording; the last window of each recording is included."""
    starts: list[int] = []
    for a, b in slices:
        if b - a >= length:
            s = list(range(a + min(offset, b - a - length), b - length + 1, stride))
            starts.extend(s if s[-1] == b - length else s + [b - length])
    return np.asarray(starts, dtype=np.int64)


def class_weights(y: np.ndarray, spec: Any) -> torch.Tensor:
    """Cross-entropy class weights: a fixed list, or AttnSleep's rule computed from TRAINING labels only
    (the repo counts train + test labels, utils/util.py calc_class_weight)."""
    if spec == "attnsleep":
        counts, mu = np.bincount(y.astype(np.int64), minlength=N_CLASSES), np.array([1.5, 2, 1.5, 1, 1.5]) / N_CLASSES
        spec = [round(max(math.log(mu[k] * counts.sum() / counts[k]), 1.0) * mu[k], 2) for k in range(N_CLASSES)]
    return torch.tensor(np.asarray(spec, dtype=np.float64), dtype=torch.float32)


def augment(x: torch.Tensor, cfg: dict[str, Any], gen: torch.Generator) -> torch.Tensor:
    """Random time shift (sequences shifted as one signal, TinySleepNet augment_signal_full) and amplitude scale."""
    if not cfg["enabled"]:
        return x
    if (max_shift := int(cfg["time_shift_frac"] * x.shape[-1])) > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), generator=gen).tolist()
        b, l, c, t = x.shape
        flat = torch.stack([torch.roll(r, s, dims=-1) for r, s in zip(x.permute(0, 2, 1, 3).reshape(b, c, l * t), shifts)])
        x = flat.reshape(b, c, l, t).permute(0, 2, 1, 3).contiguous()
    if tuple(cfg["amp_scale"]) != (1.0, 1.0):  # one factor per (sample, channel)
        x = x * torch.empty((x.shape[0], 1, x.shape[-2], 1)).uniform_(*cfg["amp_scale"], generator=gen).to(x.device, x.dtype)
    return x


def batches(d: dict[str, np.ndarray], is_seq: bool, cfg: dict[str, Any], rng: np.random.Generator) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Shuffled batches: epochs, or non-overlapping L-epoch windows with a fresh random offset every epoch."""
    bs = cfg["train"]["batch_size"]
    if not is_seq:
        perm = rng.permutation(len(d["y"]))
        for i in range(0, len(perm), bs):
            idx = np.sort(perm[i: i + bs])
            yield d["X"][idx], d["y"][idx]
        return
    length = cfg["seq"]["length"]
    starts = rng.permutation(window_starts(recording_slices(d["rec_id"]), length, length, int(rng.integers(0, length))))
    for i in range(0, len(starts), bs):
        rows = starts[i: i + bs, None] + np.arange(length)
        yield d["X"][rows], d["y"][rows]


def amp(cfg: dict[str, Any], device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg["train"]["amp_bf16"] and device.type == "cuda")


@torch.no_grad()
def predict_proba(model, d: dict[str, np.ndarray], cfg: dict[str, Any], device, stride: int) -> np.ndarray:
    """One probability row per epoch. Sequence models average each epoch over every L-window covering it
    (stride 1: context on both sides); a recording shorter than L is one shorter window."""
    model.eval()
    x, n, ebs = d["X"], len(d["X"]), cfg["train"]["eval_batch_size"]
    run = lambda xb: tF.softmax(model(torch.from_numpy(xb).to(device)).float(), dim=-1).cpu().numpy()  # noqa: E731
    if not model.is_sequence:
        with amp(cfg, device):
            return np.concatenate([run(x[i: i + ebs]) for i in range(0, n, ebs)]).astype(np.float32)
    length = cfg["seq"]["length"]
    acc, cnt = np.zeros((n, N_CLASSES)), np.zeros(n, dtype=np.int64)
    for a, b in recording_slices(d["rec_id"]):
        starts = window_starts([(a, b)], length, stride) if b - a >= length else None
        with amp(cfg, device):
            if starts is None:
                acc[a:b] += run(x[a:b][None])[0]
                cnt[a:b] += 1
                continue
            for i in range(0, len(starts), max(1, ebs // length)):
                rows = starts[i: i + max(1, ebs // length), None] + np.arange(length)
                np.add.at(acc, rows.ravel(), run(x[rows]).reshape(-1, N_CLASSES))
                np.add.at(cnt, rows.ravel(), 1)
    assert (cnt > 0).all(), "some epochs received no prediction"
    return (acc / cnt[:, None]).astype(np.float32)


def score(model, d: dict[str, np.ndarray], cfg: dict[str, Any], device, stride: int, w: torch.Tensor | None) -> dict[str, float]:
    """Accuracy, macro-F1, kappa and weighted cross-entropy of a split."""
    p, y = predict_proba(model, d, cfg, device, stride), d["y"].astype(np.int64)
    m = metrics_from_confusion(subject_confusions(y, p.argmax(1), np.zeros(len(y)))[1].sum(0))
    w = np.ones(N_CLASSES) if w is None else w.numpy()
    loss = -(w[y] * np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1.0))).sum() / w[y].sum()
    return dict(acc=float(m["acc"]), mf1=float(m["mf1"]), kappa=float(m["kappa"]), loss=float(loss))


def train_one_fold(model, tr: dict, va: dict, cfg: dict[str, Any], device, seed: int, audit: dict | None = None) -> tuple[dict, list]:
    """Early stopping on validation macro-F1; loads the best weights. ``audit`` (the test subject) is scored after
    every epoch and only logged, never used for selection: it measures AttnSleep's best-test-epoch optimism."""
    t = cfg["train"]
    w = class_weights(tr["y"], t["class_weights"])
    opt = (torch.optim.AdamW if t["optimizer"] == "adamw" else torch.optim.Adam)(
        model.parameters(), lr=t["lr"], betas=tuple(t["betas"]), eps=t["eps"], weight_decay=t["weight_decay"], amsgrad=t["amsgrad"])
    rng, gen, l2 = np.random.default_rng(seed), torch.Generator().manual_seed(seed), model.conv_l2_params() if t["l2_conv"] else []
    best, bad, hist = dict(mf1=-1.0), 0, []
    for ep in range(t["max_epochs"]):
        sch, lr = t["lr_schedule"], t["lr"]
        if sch["type"] == "step" and ep >= sch["step_epoch"]:
            lr = sch["step_lr"]  # AttnSleep: 1e-4 from epoch 10
        elif sch["type"] == "cosine":
            lr = sch["min_lr"] + 0.5 * (t["lr"] - sch["min_lr"]) * (1 + math.cos(math.pi * ep / t["max_epochs"]))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for xb, yb in batches(tr, model.is_sequence, cfg, rng):
            xb = augment(torch.from_numpy(xb).to(device), cfg["augment"], gen)
            with amp(cfg, device):
                logits = model(xb)
            loss = tF.cross_entropy(logits.float().reshape(-1, N_CLASSES), torch.from_numpy(yb.astype(np.int64)).to(device).reshape(-1),
                                    weight=w.to(device), label_smoothing=t["label_smoothing"])
            if l2:  # TinySleepNet: l2_weight_decay * tf.nn.l2_loss(w) on conv kernels
                loss = loss + t["l2_conv"] * sum((p.float() ** 2).sum() / 2 for p in l2)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if t["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        vm = score(model, va, cfg, device, cfg["seq"]["val_stride"], w)
        rec = dict(epoch=ep, lr=lr, train_loss=tot / nb, val_loss=vm["loss"], val_acc=vm["acc"], val_mf1=vm["mf1"],
                   val_kappa=vm["kappa"], sec=time.time() - t0)
        if audit is not None:
            rec.update({f"audit_test_{k}": v for k, v in score(model, audit, cfg, device, cfg["seq"]["test_stride"], None).items() if k != "loss"})
        hist.append(rec)
        if vm["mf1"] > best["mf1"]:
            best, bad = dict(mf1=vm["mf1"], epoch=ep, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}), 0
        else:
            bad += 1
        print(f"    ep {ep:3d} train_loss {rec['train_loss']:.4f} val_mf1 {vm['mf1']:.4f}", flush=True)
        if ep + 1 >= t["min_epochs"] and bad >= t["patience"]:
            break
    model.load_state_dict(best["state"])
    return dict(best_epoch=best["epoch"], best_val_mf1=best["mf1"], epochs_run=len(hist)), hist


def cv_deep(cfg: dict[str, Any], folds: Sequence[dict], seed: int, ckpt_dir: Path, out_dir: Path, eval_only: bool = False,
            data: dict[str, np.ndarray] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Subject-wise CV of one deep model and seed; eval_only re-scores the saved checkpoints."""
    from src import prepare_data as P
    from src.deep.models import build_model, count_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = P.load_dataset(sorted({s for f in folds for p in ("train", "val", "test") for s in f[f"{p}_subjects"]})) if data is None else data
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, info = [], []
    for fd in folds:
        k = fd["fold"]
        set_seeds(seed + k, cfg["train"]["deterministic"])
        tr, va, te = (prepare(data, fd[f"{p}_subjects"], cfg["data"]["normalization"]) for p in ("train", "val", "test"))
        model = build_model(cfg["model"]["name"], n_channels=1, **cfg["model"]["kwargs"]).to(device)
        summary: dict[str, Any] = {}
        if eval_only:
            model.load_state_dict(torch.load(ckpt_dir / f"fold{k}.pt", map_location=device))
        else:
            summary, hist = train_one_fold(model, tr, va, cfg, device, seed + k, te if cfg["eval"]["track_test_each_epoch"] else None)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"fold{k}.pt")
            pd.DataFrame(hist).to_csv(out_dir / f"history_fold{k}.csv", index=False)
        parts.append(predictions_frame(te, k, te["y"], predict_proba(model, te, cfg, device, cfg["seq"]["test_stride"])))
        info.append(dict(fold=k, seed=seed + k, n_params=count_params(model), **summary))
    return pd.concat(parts, ignore_index=True), dict(config=cfg, seed=seed, device=str(device), folds=info)


def run_model(experiment: str, model: str, seed: int) -> Path:
    """Train one deep model and seed of an experiment on every fold and save its run folder."""
    cfg, out = load_config(experiment)["models"][model], run_dir(experiment, model, seed)
    preds, info = cv_deep(cfg, S.load_folds(), seed, repo_path(f"checkpoints/{experiment}/{model}/seed{seed}"), out)
    save_run(out, preds, dict(experiment=experiment, model=model, **info))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="exp1_main")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    run_model(a.experiment, a.model, a.seed)
