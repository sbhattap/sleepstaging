"""YASA-style per-epoch features of EEG Fpz-Cz for LightGBM (configs/base.yaml `features:`).

YASA (Vallat & Walker 2021; yasa 0.8.1 SleepStaging) plus log absolute band powers, spectral and sample entropy
and a +-2 epoch context window. Computed per recording on the WHOLE untrimmed night, so rolling windows and
per-recording scaling never cross nights and use no labels; the protocol filter is applied when loading.

CLI:  uv run python -m src.classical.features      # -> data/processed/features_fpz.parquet (~5 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src import prepare_data as P
from src.config import load_config, repo_path

META_COLS = ["subject", "night", "rec_id", "epoch_idx", "label", "in_trim_window"]
TIME_COLS = ["time_hour", "time_norm"]
CONTEXT_SUFFIXES = ("_c7min_norm", "_p2min_norm", "_c2ep_norm")  # 7.5-min centred, 2-min past, +-2 epochs


def epoch_features(ep: np.ndarray, sf: float, cfg: dict[str, Any]) -> pd.DataFrame:
    """Features of band-passed epochs (n, samples): statistics, Hjorth, band powers and ratios, entropies, fractals."""
    import antropy as ant
    from scipy import signal
    from scipy.integrate import trapezoid
    from yasa import bandpower_from_psd_ndarray

    bands = [(lo, hi, name) for name, (lo, hi) in cfg["bands"].items()]
    with np.errstate(invalid="ignore", divide="ignore"):
        hmob, hcomp = ant.hjorth_params(ep, axis=1)
        q, dev = np.quantile(ep, [0.25, 0.75], axis=1), ep - ep.mean(axis=1, keepdims=True)
        m2 = (dev ** 2).mean(axis=1)
        f = dict(std=np.std(ep, ddof=1, axis=1), iqr=q[1] - q[0], skew=(dev ** 3).mean(axis=1) / m2 ** 1.5,
                 kurt=(dev ** 4).mean(axis=1) / m2 ** 2 - 3, nzc=ant.num_zerocross(ep, axis=1).astype(np.float64), hmob=hmob, hcomp=hcomp)
        w = cfg["welch"]
        freqs, psd = signal.welch(ep, sf, window=w["window"], nperseg=int(w["win_sec"] * sf), average=w["average"])
        rel, ab = (bandpower_from_psd_ndarray(psd, freqs, bands=bands, relative=r) for r in (True, False))
        for j, (_, _, b) in enumerate(bands):
            f[b], f[f"{b}_abs"] = rel[j], np.log10(ab[j] + 1e-12)  # log10 µV² (ours); the floor keeps flat epochs finite
        delta = f["sdelta"] + f["fdelta"]
        f.update(dt=delta / f["theta"], ds=delta / f["sigma"], db=delta / f["beta"], at=f["alpha"] / f["theta"])
        broad = (freqs >= cfg["filter"]["l_freq"]) & (freqs <= cfg["filter"]["h_freq"])
        p = psd[:, broad] / np.clip(psd[:, broad].sum(axis=1, keepdims=True), 1e-30, None)
        f["abspow"] = trapezoid(psd[:, broad], dx=freqs[1] - freqs[0])
        f["spec_ent"] = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1) / np.log2(broad.sum())
        pe = cfg["perm_entropy"]
        f["perm"] = np.array([ant.perm_entropy(e, order=pe["order"], delay=pe["delay"], normalize=True) for e in ep])
        f["higuchi"] = np.array([ant.higuchi_fd(e) for e in ep])
        f["petrosian"] = ant.petrosian_fd(ep, axis=1)
        f["sampen"] = np.array([ant.sample_entropy(e, order=cfg["sample_entropy_order"]) for e in ep])
    return pd.DataFrame(f)


def signal_features(x: np.ndarray, sf: float, cfg: dict[str, Any]) -> pd.DataFrame:
    """One night of contiguous epochs x (n, samples) -> raw, per-night robust-scaled, context and time features."""
    from mne.filter import filter_data
    from sklearn.preprocessing import robust_scale

    # Band-pass the night as one continuous signal (no edge transients at every 30-s boundary), as YASA does.
    ep = filter_data(x.reshape(-1).astype(np.float64), sf, cfg["filter"]["l_freq"], cfg["filter"]["h_freq"], verbose=False).reshape(x.shape)
    raw = epoch_features(ep, sf, cfg).add_prefix("eeg_").replace([np.inf, -np.inf], np.nan).fillna(0.0)  # flat epochs -> 0
    c = cfg["context"]
    scale = lambda d: pd.DataFrame(robust_scale(d.to_numpy(np.float64), quantile_range=tuple(c["robust_quantile_range"])),  # noqa: E731
                                   columns=d.columns, index=d.index)
    windows = (raw.rolling(c["centered_triang_epochs"], center=True, min_periods=1, win_type="triang").mean(),  # YASA
               raw.rolling(c["past_epochs"], min_periods=1).mean(),                                              # YASA
               raw.rolling(c["short_centered_epochs"], center=True, min_periods=1).mean())                       # ours
    t = np.arange(len(raw)) * x.shape[1] / sf
    return pd.concat([raw, scale(raw).add_suffix("_norm")] + [scale(w).add_suffix(s) for w, s in zip(windows, CONTEXT_SUFFIXES)]
                     + [pd.DataFrame({"time_hour": t / 3600, "time_norm": t / t[-1]})], axis=1).astype(np.float32)


def recording_features(args: tuple[int, int, dict[str, Any]]) -> pd.DataFrame:
    """Feature table of one recording, with ids, label and trim mask first."""
    s, n, cfg = args
    r = P.load_recording(s, n)
    assert np.allclose(np.diff(r["epoch_onset_s"]), 30.0), "non-contiguous epochs"
    f = signal_features(r["x"][:, r["ch_names"].index(cfg["channel"])], r["sfreq"], cfg)
    meta = pd.DataFrame({"subject": np.full(len(f), s, np.int16), "night": np.full(len(f), n, np.int8),
                         "rec_id": np.full(len(f), s * 10 + n, np.int32), "epoch_idx": np.arange(len(f), dtype=np.int32),
                         "label": r["y"].astype(np.int8), "in_trim_window": r["in_trim_window"].astype(bool)})
    return pd.concat([meta, f], axis=1)


def feature_columns(columns: Sequence[str], context: bool) -> list[str]:
    """Model inputs: current-epoch features (raw + _norm); context=True adds the rolling windows and time of night."""
    return [c for c in columns if c not in META_COLS and (context or (c not in TIME_COLS and not c.endswith(CONTEXT_SUFFIXES)))]


def load_features(protocol: bool = True) -> pd.DataFrame:
    """The cached feature table (built on first use); protocol=True keeps the AttnSleep epochs only."""
    base = load_config()
    path = repo_path(base["paths"]["features_file"])
    if not path.exists():
        with ProcessPoolExecutor(base["features"]["workers"]) as ex:
            pairs = P.available_recordings(range(*base["data"]["subjects"]))
            df = pd.concat(list(ex.map(recording_features, [(s, n, base["features"]) for s, n in pairs])), ignore_index=True)
        assert len(pairs) == base["data"]["n_recordings"] and np.isfinite(df.drop(columns=META_COLS).to_numpy()).all()
        df.to_parquet(path, index=False)
    df = pd.read_parquet(path)
    return df[df["in_trim_window"] & (df["label"] >= 0)].reset_index(drop=True) if protocol else df


if __name__ == "__main__":
    print(load_features(protocol=False).shape)
