"""Deep models: output shapes, sliding-window averaging, and a tiny end-to-end CPU run (train, then re-score)."""

import copy

import numpy as np
import pytest
import torch

from src.config import load_config
from src.deep import train as tr
from src.deep.models import build_model


@pytest.mark.parametrize("name", ["attnsleep", "tinysleepnet"])
def test_forward_shapes(name):
    m = build_model(name).eval()
    with torch.no_grad():
        out = m(torch.randn(3, 5, 1, 3000) if m.is_sequence else torch.randn(3, 1, 3000))
    assert out.shape == ((3, 5, 5) if m.is_sequence else (3, 5)) and torch.isfinite(out).all()


class _Toy(torch.nn.Module):
    is_sequence = True

    def forward(self, x):  # output depends on content and position, so wrong window averaging would show
        v = x[..., 0, 0]
        return v[..., None] * torch.arange(1.0, 6.0) + 0.3 * torch.arange(v.shape[1])[:, None] * torch.arange(5.0)


def test_each_epoch_is_the_mean_over_its_windows():
    L, rec = 4, np.array([1] * 9 + [2] * 2 + [3] * 6)
    d = {"X": np.random.default_rng(0).normal(size=(len(rec), 1, 3)).astype(np.float32), "rec_id": rec}
    got = tr.predict_proba(_Toy(), d, {"seq": {"length": L}, "train": {"eval_batch_size": 8, "amp_bf16": False}}, torch.device("cpu"), 1)
    acc, cnt = np.zeros((len(rec), 5)), np.zeros(len(rec))
    for a, b in tr.recording_slices(rec):
        for s in ([a] if b - a < L else range(a, b - L + 1)):
            e = min(s + L, b)
            acc[s:e] += torch.softmax(_Toy()(torch.from_numpy(d["X"][s:e][None])), -1)[0].numpy()
            cnt[s:e] += 1
    np.testing.assert_allclose(got, acc / cnt[:, None], atol=1e-5)


@pytest.mark.parametrize("model", ["tinysleepnet", "attnsleep"])
def test_cv_deep_end_to_end_cpu(model, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = copy.deepcopy(load_config("exp1_main")["models"][model])
    cfg["seq"].update(length=4, val_stride=2)
    cfg["train"].update(batch_size=8, eval_batch_size=32, max_epochs=2, min_epochs=1, patience=1, amp_bf16=False)
    rng, t = np.random.default_rng(0), np.arange(3000) / 100.0
    y = rng.permutation(np.repeat(np.arange(5), 32)).astype(np.int8)  # 4 subjects x 40 epochs, class-dependent sines
    x = 30 * np.sin(2 * np.pi * np.array([10.0, 6.0, 13.0, 1.5, 4.0])[y][:, None] * t) + rng.normal(0, 5, (160, 3000))
    data = dict(X=x[:, None].astype(np.float32), y=y, subject=np.repeat(np.arange(4), 40), night=np.ones(160, int),
                rec_id=np.repeat(np.arange(4) * 10 + 1, 40), epoch_idx=np.tile(np.arange(40), 4),
                rec_center=np.zeros((160, 1), np.float32), rec_scale=np.full((160, 1), 30.0, np.float32))
    folds = [dict(fold=k, test_subjects=[k], val_subjects=[(k + 1) % 4], train_subjects=[s for s in range(4) if s not in (k, (k + 1) % 4)])
             for k in range(2)]
    preds, _ = tr.cv_deep(cfg, folds, 42, tmp_path / "ck", tmp_path / "out", data=data)
    again, _ = tr.cv_deep(cfg, folds, 42, tmp_path / "ck", tmp_path / "out2", eval_only=True, data=data)
    assert len(preds) == 80 and np.allclose(preds.filter(like="p_").sum(axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(again.filter(like="p_").to_numpy(), preds.filter(like="p_").to_numpy(), atol=1e-6)
