"""Sleep-EDF-20: raw EDF -> 30-s epochs -> cleaning checks -> missingness diagnostics (configs/base.yaml `data:`).

1. epoch    per recording: verify SHA256, validate the EDF header, align the hypnogram, map R&K -> AASM,
            cut 30-s epochs, store the wake-trim mask, save data/processed/SC4<ss><n>.npz, delete the raw EDFs
2. clean    schema, labels, duplicates, artifact flags (columns, never deletions), exclusion flow, final counts
3. missing  what is missing and what predicts it (evidence for MCAR / MAR / MNAR)
Tables go to data/reports/ (tracked). Run `bash data/download_sleepedf20.sh` first.

CLI:  uv run python -m src.prepare_data
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import load_config, repo_path

STAGE = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def rec_name(subject: int, night: int) -> str:
    """Recording stem, e.g. subject 3 night 2 -> 'SC4032'."""
    return f"SC4{subject:02d}{night}"


# ============================================================================ 1. epoch one recording
def recording_table(base: dict[str, Any]) -> pd.DataFrame:
    """One row per recording: file names, sizes and SHA256 from PhysioNet's listing (cached by the download script).

    Hypnogram names differ from PSG names (the 8th letter is the scoring technician), so they are read, not derived.
    """
    src = base["data"]["source"]
    sums = {p.split("/")[1]: h for h, p in (l.split() for l in repo_path(src["checksum_cache"]).read_text().splitlines()
                                            if " sleep-cassette/" in l)}
    html = repo_path(src["index_cache"])
    if not html.exists():
        import urllib.request
        html.write_bytes(urllib.request.urlopen(src["base_url"] + src["cassette_dir"], timeout=60).read())
    rows = [dict(fname=f, bytes=int(b), subject=int(f[3:5]), night=int(f[5]), kind=f.split("-")[1][:3].lower())
            for f, b in re.findall(r'href="(SC4\d{3}\w{2}-(?:PSG|Hypnogram)\.edf)">[^<]*</a>\s+\S+\s+\S+\s+(\d+)', html.read_text())]
    t = pd.DataFrame(rows).query("@base['data']['subjects'][0] <= subject < @base['data']['subjects'][1]")
    t = t.pivot(index=["subject", "night"], columns="kind", values=["fname", "bytes"])
    t.columns = [f"{k}_{'file' if v == 'fname' else 'bytes'}" for v, k in t.columns]
    t = t.reset_index().astype({"psg_bytes": int, "hyp_bytes": int})
    t["psg_sha256"], t["hyp_sha256"] = t.psg_file.map(sums), t.hyp_file.map(sums)
    # Integrity: one PSG + one hypnogram per night, a checksum for each, 39 nights, no file over 2 GB.
    assert t.notna().all().all() and len(t) == base["data"]["n_recordings"], "incomplete PhysioNet file set"
    assert (t.psg_bytes < src["max_file_bytes"]).all(), "a file exceeds the 2 GB rule"
    return t


def read_edf_header(path: Path) -> dict[str, Any]:
    """Parse the fixed-format EDF header ourselves, so validation does not depend on a reader's conversions."""
    with open(path, "rb") as f:
        head = f.read(256)
        ns = int(head[252:256])
        sig = f.read(256 * ns)
    field = lambda off, w: [sig[off * ns + i * w: off * ns + (i + 1) * w].decode("latin-1").strip() for i in range(ns)]  # noqa: E731
    dd, mm, yy = (int(v) for v in head[168:176].decode().split("."))
    hh, mi, ss = (int(v) for v in head[176:184].decode().split("."))
    return dict(start=datetime(1900 + yy if yy >= 85 else 2000 + yy, mm, dd, hh, mi, ss), n_records=int(head[236:244]),
                record_sec=float(head[244:252]), labels=field(0, 16), units=field(96, 8),
                phys_min=[float(v) for v in field(104, 8)], phys_max=[float(v) for v in field(112, 8)],
                samples_per_record=[int(v) for v in field(216, 8)])


def hypnogram_epochs(onsets: np.ndarray, durations: np.ndarray, descs: Sequence[str], offset_s: float,
                     signal_sec: float, label_map: dict[str, int], epoch_sec: int) -> dict[str, Any]:
    """Expand (onset, duration, label) annotations into 30-s epochs; drop and count those past the signal end."""
    ep_on, ep_raw, anomalies = [], [], []
    for on, dur, d in zip(onsets, durations, descs):
        assert d in label_map, f"unknown hypnogram label {d!r}"
        on = float(on) + offset_s
        if on % epoch_sec or dur % epoch_sec:
            anomalies.append(dict(onset=on, duration=float(dur), label=d))  # off the 30-s grid: recorded, floored
        n = int(dur // epoch_sec)
        ep_on.extend(on + epoch_sec * np.arange(n))
        ep_raw.extend([d] * n)
    ep_on, ep_raw = np.asarray(ep_on, dtype=np.float64), np.asarray(ep_raw)
    assert np.all(np.diff(ep_on) > 0), "hypnogram epochs overlap or are out of order"
    inside = (ep_on >= 0) & (ep_on + epoch_sec <= signal_sec)
    return dict(onset=ep_on[inside], y_raw=ep_raw[inside], y=np.array([label_map[d] for d in ep_raw[inside]], dtype=np.int8),
                n_hyp_epochs=len(ep_on), n_outside=int((~inside).sum()), anomalies=anomalies,
                outside_counts={k: int(v) for k, v in zip(*np.unique(ep_raw[~inside], return_counts=True))},
                hyp_end_s=float(ep_on[-1] + epoch_sec))


def trim_mask(y: np.ndarray, edge_epochs: int) -> np.ndarray:
    """DeepSleepNet / AttnSleep wake trim: drop Movement/'?' first, keep [first non-W - edge, last non-W + edge]
    on that shorter sequence (clipped), then map the window back to the original positions."""
    valid = np.flatnonzero(y >= 0)
    mask = np.zeros(len(y), dtype=bool)
    nw = np.flatnonzero(y[valid] != 0)
    if len(nw):
        mask[valid[max(nw[0] - edge_epochs, 0)]: valid[min(nw[-1] + edge_epochs, len(valid) - 1)] + 1] = True
    return mask


def epoch_signal(data: np.ndarray, onsets_s: np.ndarray, sfreq: float, epoch_sec: int) -> np.ndarray:
    """Cut (n_channels, n_samples) into (n_epochs, n_channels, epoch_sec * sfreq) at the given onsets."""
    n, starts = int(round(epoch_sec * sfreq)), np.round(onsets_s * sfreq).astype(np.int64)
    assert np.all(starts >= 0) and np.all(starts + n <= data.shape[1]), "epoch outside signal"
    return np.ascontiguousarray(data[:, starts[:, None] + np.arange(n)].transpose(1, 0, 2))


def process_recording(row: dict[str, Any], base: dict[str, Any]) -> str:
    """Validate, epoch and save one recording from data/raw/; delete its raw EDFs afterwards. Returns a status."""
    import mne

    cfg, sig, s, n = base["data"], base["data"]["signal"], int(row["subject"]), int(row["night"])
    out = repo_path(base["paths"]["processed_dir"]) / f"{rec_name(s, n)}.npz"
    if out.exists():
        return "exists"
    raw_dir = repo_path(base["paths"]["raw_dir"]) / rec_name(s, n)
    psg, hyp = raw_dir / row["psg_file"], raw_dir / row["hyp_file"]
    for path, sha in ((psg, row["psg_sha256"]), (hyp, row["hyp_sha256"])):
        assert path.exists(), f"{path} missing: run `bash data/download_sleepedf20.sh`"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, f"{path}: SHA256 mismatch"
    h, h_hyp = read_edf_header(psg), read_edf_header(hyp)
    info = {}
    for ch in sig["channels"] + [sig["emg_channel"]]:  # fail loudly on a missing channel, rate or unit
        i = h["labels"].index(ch)
        info[ch] = dict(sfreq=h["samples_per_record"][i] / h["record_sec"], unit=h["units"][i],
                        phys_min=h["phys_min"][i], phys_max=h["phys_max"][i])
        assert ch == sig["emg_channel"] or (info[ch]["sfreq"] == sig["sfreq"] and info[ch]["unit"] in ("uV", "µV")), info[ch]
    signal_sec = h["n_records"] * h["record_sec"]
    offset_s = (h_hyp["start"] - h["start"]).total_seconds()  # 0 in every SC file; applied anyway
    ann = mne.read_annotations(hyp)
    hy = hypnogram_epochs(ann.onset, ann.duration, list(ann.description), offset_s, signal_sec, cfg["labels"]["map"], sig["epoch_sec"])
    data = mne.io.read_raw_edf(psg, include=sig["channels"], preload=True, verbose="error").get_data(units="uV").astype(np.float32)
    emg = mne.io.read_raw_edf(psg, include=[sig["emg_channel"]], preload=True, verbose="error")  # own 1-Hz rate
    emg_fs, emg = float(emg.info["sfreq"]), emg.get_data(units="uV")[0]
    assert data.shape[1] == int(round(signal_sec * sig["sfreq"])), "signal length != header length"
    x = epoch_signal(data, hy["onset"], sig["sfreq"], sig["epoch_sec"])
    assert np.isfinite(x).all()
    k, e0 = int(sig["epoch_sec"] * emg_fs), np.round(hy["onset"] * emg_fs).astype(int)
    emg_mean = np.array([emg[a: a + k].mean() if a + k <= len(emg) else np.nan for a in e0], dtype=np.float32)
    meta = dict(psg_file=row["psg_file"], hyp_file=row["hyp_file"], psg_sha256=row["psg_sha256"], hyp_sha256=row["hyp_sha256"],
                psg_start=h["start"].isoformat(), hyp_start=h_hyp["start"].isoformat(), offset_s=offset_s,
                signal_sec=signal_sec, hyp_end_s=hy["hyp_end_s"], n_hyp_epochs=hy["n_hyp_epochs"],
                n_outside_signal=hy["n_outside"], outside_counts=hy["outside_counts"], anomalies=hy["anomalies"],
                channel_info=info, emg_sfreq=emg_fs, all_labels=h["labels"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, x=x, ch_names=np.array(sig["channels"]), y=hy["y"], y_raw=hy["y_raw"], subject=np.int64(s),
             night=np.int64(n), sfreq=np.float64(sig["sfreq"]), epoch_onset_s=hy["onset"],
             in_trim_window=trim_mask(hy["y"], int(cfg["trim"]["wake_edge_min"] * 60 // sig["epoch_sec"])),
             emg_epoch_mean=emg_mean, phys_min=np.array([info[c]["phys_min"] for c in sig["channels"]]),
             phys_max=np.array([info[c]["phys_max"] for c in sig["channels"]]), meta_json=np.array(json.dumps(meta)))
    tmp.rename(out)
    shutil.rmtree(raw_dir)  # keeps raw disk use near one recording per worker
    return "processed"


# ============================================================================ loading (used by the models)
def load_recording(subject: int, night: int, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """One processed recording as a dict of arrays (data/DATA_DICTIONARY.md section 3)."""
    f = repo_path((base or load_config())["paths"]["processed_dir"]) / f"{rec_name(subject, night)}.npz"
    with np.load(f) as z:
        out = {k: z[k] for k in z.files}
    out.update(subject=int(out["subject"]), night=int(out["night"]), sfreq=float(out["sfreq"]), ch_names=[str(c) for c in out["ch_names"]])
    return out


def available_recordings(subjects: Iterable[int], base: dict[str, Any] | None = None) -> list[tuple[int, int]]:
    """(subject, night) pairs that have a processed file, sorted."""
    files = repo_path((base or load_config())["paths"]["processed_dir"]).glob("SC4*.npz")
    pairs = [(int(f.stem[3:5]), int(f.stem[5])) for f in files if re.fullmatch(r"SC4\d{3}", f.stem)]
    return sorted(p for p in pairs if p[0] in set(subjects))


def recording_robust_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel median and IQR / 1.349 (= SD for Gaussian data) over a whole recording, robust to artifacts."""
    lo, med, hi = np.percentile(x.transpose(1, 0, 2).reshape(x.shape[1], -1), [25.0, 50.0, 75.0], axis=1)
    assert np.all(hi > lo), "zero-scale channel"
    return med.astype(np.float32), ((hi - lo) / 1.349).astype(np.float32)


def load_dataset(subjects: Iterable[int], channels: Sequence[str] = ("EEG Fpz-Cz",), trim: bool = True,
                 drop_unscored: bool = True, base: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Concatenated recordings; trim + drop_unscored = the AttnSleep protocol epochs.

    Returns X (N, C, 3000), y, subject, night, rec_id (subject * 10 + night), epoch_idx (position in the .npz) and
    rec_center / rec_scale: per-recording channel statistics over the whole night, so they use no labels.
    Note: the trim window and Movement/'?' removal use LABELS, on test nights too (the published protocol).
    """
    pairs = available_recordings(subjects, base)
    assert {s for s, _ in pairs} == set(subjects), "some subjects are not processed"
    parts: dict[str, list] = {k: [] for k in ("X", "y", "subject", "night", "rec_id", "epoch_idx", "rec_center", "rec_scale")}
    for s, n in pairs:
        r = load_recording(s, n, base)
        x = r["x"][:, [r["ch_names"].index(c) for c in channels], :]
        idx = np.flatnonzero((r["in_trim_window"] | (not trim)) & ((r["y"] >= 0) | (not drop_unscored)))
        center, scale = recording_robust_stats(x)  # computed BEFORE the label-based selection
        for k, v in dict(X=x[idx], y=r["y"][idx], subject=np.full(len(idx), s, np.int16), night=np.full(len(idx), n, np.int8),
                         rec_id=np.full(len(idx), s * 10 + n, np.int32), epoch_idx=idx.astype(np.int32),
                         rec_center=np.repeat(center[None], len(idx), 0), rec_scale=np.repeat(scale[None], len(idx), 0)).items():
            parts[k].append(v)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    assert not drop_unscored or set(np.unique(out["y"])) <= set(STAGE), "unscored labels leaked into the dataset"
    return out


# ============================================================================ 2. clean
def qc_flags(x: np.ndarray, phys_min: float, phys_max: float, sfreq: float, q: dict[str, Any], ref: np.ndarray) -> pd.DataFrame:
    """Per-epoch QC stats and flat / clipped / high_amp flags for one channel (x: (n, 3000) µV).

    High amplitude = robust z of log(SD) > 3, with median/MAD over the recording's scored in-window epochs
    (so ~14 h of daytime wake does not set the baseline; the epoch's own stage is never used).
    """
    tol, sd = q["clip_tol_frac"] * (phys_max - phys_min), x.std(1)
    same = np.diff(x, axis=1) == 0  # longest run of identical consecutive samples
    cur, run = np.zeros(len(x), np.int32), np.zeros(len(x), np.int32)
    for j in range(same.shape[1]):
        cur = (cur + 1) * same[:, j]
        np.maximum(run, cur, out=run)
    log_sd = np.log(np.maximum(sd, 1e-3))
    med = np.median(log_sd[ref])
    mad = 1.4826 * np.median(np.abs(log_sd[ref] - med))  # 1.4826: MAD -> SD under normality
    st = pd.DataFrame(dict(sd=sd, ptp=np.ptp(x, 1), max_abs=np.abs(x).max(1), flat_run_s=np.where(run > 0, run + 1, 1) / sfreq,
                           clip_frac=((x >= phys_max - tol) | (x <= phys_min + tol)).mean(1), log_sd_z=(log_sd - med) / mad))
    st["flat"] = (st.sd < q["flat_std_uv"]) | (st.flat_run_s >= q["flat_run_s"])
    st["clipped"], st["high_amp"] = st.clip_frac >= q["clip_frac"], st.log_sd_z > q["high_amp_robust_z"]
    return st


def recording_flags(rec: dict[str, Any], cfg: dict[str, Any], meta: pd.DataFrame) -> pd.DataFrame:
    """Schema / label / ID checks (fail loudly), then one row per epoch: ids, label, QC stats and flags per channel.

    flat / clipped / high_amp without suffix are Fpz-Cz (the model channel); *_any = any of the 3 stored channels.
    """
    m, sig, lm = json.loads(str(rec["meta_json"])), cfg["signal"], cfg["labels"]["map"]
    assert rec["x"].shape[1:] == (3, 3000) and rec["x"].dtype == np.float32 and rec["ch_names"] == sig["channels"]
    assert m["offset_s"] == 0 and not m["anomalies"] and np.all(np.diff(rec["epoch_onset_s"]) > 0)
    assert np.array_equal(np.array([lm[str(v)] for v in rec["y_raw"]], dtype=np.int8), rec["y"]), "y != map(y_raw)"
    assert m["psg_file"][:6] == rec_name(rec["subject"], rec["night"]) and len(meta.query("subject == @rec['subject'] and night == @rec['night']"))
    df = pd.DataFrame(dict(subject=rec["subject"], night=rec["night"], epoch_idx=np.arange(len(rec["y"])), label=rec["y"],
                           label_raw=rec["y_raw"], onset_s=rec["epoch_onset_s"], in_trim_window=rec["in_trim_window"],
                           emg_mean=rec["emg_epoch_mean"]))
    ref, flags = rec["in_trim_window"] & (rec["y"] >= 0), {}
    for ci, sfx in enumerate(("fpz", "pz", "eog")):
        st = qc_flags(rec["x"][:, ci], rec["phys_min"][ci], rec["phys_max"][ci], rec["sfreq"], cfg["qc"], ref)
        for c in st.columns:
            df[f"{c}_{sfx}"] = st[c].to_numpy()
        for c in ("flat", "clipped", "high_amp"):
            flags.setdefault(c, []).append(st[c].to_numpy())
    for c, arrs in flags.items():
        df[c], df[f"{c}_any"] = df[f"{c}_fpz"], np.logical_or.reduce(arrs)
    a, b = (v - v.mean(1, keepdims=True) for v in (rec["x"][:-1, 0].astype(np.float64), rec["x"][1:, 0].astype(np.float64)))
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))  # adjacent-epoch correlation: r near 1 = copied / frozen data
    df["adj_corr_fpz"] = np.append(np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan), np.nan)
    df["hash"] = [hashlib.blake2b(e.tobytes(), digest_size=8).hexdigest() for e in rec["x"]]
    return df


def exclusion_flow(flags: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """CONSORT-style table in DeepSleepNet's order (past signal end -> '?' -> Movement -> wake trim).

    Artifact flags are counted, not removed. Per-class columns give the counts after each step.
    """
    rows, cur = [], flags
    start, outside = int(manifest.n_hyp_epochs.sum()), int(manifest.n_outside_signal.sum())

    def add(step: str, removed: int, note: str = "") -> None:
        counts = {v: int((cur.label == k).sum()) for k, v in STAGE.items()}
        rows.append(dict(step=step, removed=removed, remaining=len(cur), note=note, **counts))

    rows.append(dict(step="hypnogram epochs (all recordings)", removed=0, remaining=start, note=f"{len(manifest)} recordings"))
    assert len(cur) == start - outside
    add("drop epochs past end of signal (truncated tail)", outside, "all '?' in Sleep-EDF-20")
    for keep, step in ((cur.label != -2, "drop Unknown '?' epochs inside signal"), (None, "drop Movement time epochs"),
                       (None, "drop wake more than 30 min before/after sleep (DeepSleepNet / AttnSleep trim)")):
        n0 = len(cur)
        cur = cur[keep] if keep is not None else cur[cur.label != -1] if "Movement" in step else cur[cur.in_trim_window]
        add(step, n0 - len(cur))
    add("FINAL (scored epochs used for training and scoring)", 0, "N3 includes R&K stage 4")
    nflag = {c: int(cur[c].sum()) for c in ("flat", "clipped", "high_amp")}
    add("info: Fpz-Cz flat / clipped / high_amp flagged (kept, not removed)", 0,
        json.dumps(nflag) + f"; any flag: {int((cur.flat | cur.clipped | cur.high_amp).sum())}")
    return pd.DataFrame(rows)


# ============================================================================ 3. missingness
def add_context(flags: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    """Add nearest scored stage before / after, hours since lights-off and missing indicators to the flag table."""
    out = []
    for (s, n), g in flags.groupby(["subject", "night"], sort=True):
        g, lab = g.sort_values("epoch_idx").copy(), g.sort_values("epoch_idx").label.to_numpy()
        for col, order in (("prev_stage", range(len(lab))), ("next_stage", range(len(lab) - 1, -1, -1))):
            vals, last = np.full(len(lab), "none", dtype=object), "none"
            for i in order:
                vals[i] = last
                last = STAGE[int(lab[i])] if lab[i] >= 0 else last
            g[col] = vals
        m = man.query("subject == @s and night == @n").iloc[0]
        t0, (hh, mm, ss) = datetime.fromisoformat(m.psg_start), (int(v) for v in m.lights_off.split(":"))
        lights = ((hh * 3600 + mm * 60 + ss) - (t0.hour * 3600 + t0.minute * 60 + t0.second)) % 86400
        g["hours"] = (g.onset_s - lights) / 3600
        scored = np.flatnonzero(lab >= 0)
        g["edge"] = (np.arange(len(g)) < scored[0]) | (np.arange(len(g)) > scored[-1])  # leading / trailing unscored block
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df["missing"], df["is_move"], df["is_unk"] = (df.label < 0).astype(int), (df.label == -1).astype(int), (df.label == -2).astype(int)
    df["artifact"] = (df.flat | df.clipped | df.high_amp).astype(int)
    return df


def block_tests(df: pd.DataFrame, outcome: str, blocks: dict[str, str], hours_bin: float) -> pd.DataFrame:
    """Likelihood-ratio test for dropping each block from the logistic model ``outcome ~ all blocks``.

    Epochs are collapsed to covariate cells with counts (time binned): same likelihood, ~100x smaller fit.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats

    cols = [c for c in ("prev_stage", "next_stage", "stage", "subject", "hours") if any(re.search(rf"\b{c}\b", t) for t in blocks.values())]
    d = df[cols + [outcome]].assign(hours=lambda t: (t.hours / hours_bin).round() * hours_bin)
    d = d.groupby(cols + [outcome], observed=True).size().rename("w").reset_index()
    with warnings.catch_warnings():  # subject dummies with zero events separate; their likelihood still converges
        warnings.simplefilter("ignore")
        fit = lambda f: smf.glm(f, d, family=sm.families.Binomial(), freq_weights=d["w"]).fit(maxiter=50)  # noqa: E731
        full, rows = fit(f"{outcome} ~ " + " + ".join(blocks.values())), []
        for name in blocks:
            red = fit(f"{outcome} ~ " + (" + ".join(t for k, t in blocks.items() if k != name) or "1"))
            lr, ddf = 2 * (full.llf - red.llf), int(full.df_model - red.df_model)
            rows.append(dict(outcome=outcome, term=name, lr_chi2=round(float(lr), 2), df=ddf, p_value=float(stats.chi2.sf(lr, ddf))))
    return pd.DataFrame(rows)


def missingness_tables(df: pd.DataFrame, meta: pd.DataFrame, hours_bin: float, rep: Path) -> dict[str, Any]:
    """Write the missingness diagnostics to data/reports/ and return the scalar test results."""
    from scipy import stats

    per = meta.groupby("subject").agg(n_nights=("night", "nunique"), age=("age", "first"), sex=("sex", "first"))
    per.to_csv(rep / "missingness_nights_per_subject.csv")  # whole SC release: does losing a night depend on age / sex?
    one, two = per[per.n_nights == 1], per[per.n_nights == 2]
    facts = dict(single_night_subjects_sc_release=one.index.tolist(),
                 nights_age_mannwhitney_p=float(stats.mannwhitneyu(one.age, two.age).pvalue),
                 nights_sex_fisher_p=float(stats.fisher_exact(pd.crosstab(per.n_nights == 1, per.sex).to_numpy())[1]))
    win = df[df.in_trim_window]  # Movement / '?' inside the scored window: what predicts them?
    by = win.groupby("subject")[["is_move", "is_unk"]].sum().assign(n=win.groupby("subject").size())
    by.assign(rate_pct=100 * (by.is_move + by.is_unk) / by.n).to_csv(rep / "missingness_unscored_by_subject.csv")
    blocks = {"prev_stage": "C(prev_stage)", "next_stage": "C(next_stage)", "time_of_night": "hours + I(hours**2)", "subject": "C(subject)"}
    both = (df.prev_stage != "none") & (df.next_stage != "none")
    block_tests(win[both[win.index]], "missing", blocks, hours_bin).to_csv(rep / "missingness_logit_unscored.csv", index=False)
    block_tests(df[~df.edge & both], "is_unk", blocks, hours_bin).to_csv(rep / "missingness_logit_unknown_fullnight.csv", index=False)
    facts.update(unscored_in_window=int(win.missing.sum()), movement_in_window=int(win.is_move.sum()), unk_in_window=int(win.is_unk.sum()),
                 epochs_in_window=len(win), unk_edge=int(df[df.edge].is_unk.sum()), unk_interior=int(df[~df.edge].is_unk.sum()))
    sc = df[df.in_trim_window & (df.label >= 0)].assign(stage=lambda t: t.label.map(STAGE))  # artifacts vs stage
    cols = ["flat", "clipped", "high_amp", "artifact", "flat_any", "clipped_any", "high_amp_any"]
    tab = (sc.groupby("stage")[cols].mean() * 100).T[list(STAGE.values())]
    tab.assign(all=100 * sc[cols].mean()).to_csv(rep / "artifact_rate_by_stage.csv")
    sc.groupby("stage")[cols].sum().T[list(STAGE.values())].to_csv(rep / "artifact_count_by_stage.csv")
    block_tests(sc, "artifact", {"stage": "C(stage)", "time_of_night": "hours + I(hours**2)", "subject": "C(subject)"},
                hours_bin).to_csv(rep / "artifact_logit.csv", index=False)
    art, clean = sc[sc.artifact == 1].emg_mean.dropna(), sc[sc.artifact == 0].emg_mean.dropna()
    facts.update(artifact_epochs=int(sc.artifact.sum()), emg_median_artifact=float(art.median()), emg_median_clean=float(clean.median()),
                 emg_mannwhitney_p=float(stats.mannwhitneyu(art, clean).pvalue))
    return facts


# ============================================================================ driver
def subject_metadata(base: dict[str, Any]) -> pd.DataFrame:
    """Age, sex and lights-off per (subject, night) for the whole SC release (MNE's copy of SC-subjects.xls)."""
    import mne

    df = pd.read_csv(Path(mne.__file__).parent / base["data"]["source"]["metadata_csv_in_mne"])
    return df[df["record type"] == "PSG"][["subject", "night", "age", "sex", "lights off"]].rename(
        columns={"lights off": "lights_off"}).reset_index(drop=True)


def _process(args: tuple[dict[str, Any], dict[str, Any]]) -> str:
    return process_recording(*args)


def run() -> None:
    """All steps for Sleep-EDF-20; any failed check stops the run."""
    base = load_config()
    cfg, rep = base["data"], repo_path(base["paths"]["reports_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    tab = recording_table(base)
    tab.to_csv(rep / "physionet_files.csv", index=False)
    with ProcessPoolExecutor(max_workers=cfg["processing"]["n_workers"]) as ex:
        list(ex.map(_process, [(r, base) for r in tab.to_dict("records")]))
    meta = subject_metadata(base)
    meta.query(f"{cfg['subjects'][0]} <= subject < {cfg['subjects'][1]}").to_csv(rep / "subject_metadata.csv", index=False)
    rows, tables = [], []
    for s, n in available_recordings(range(*cfg["subjects"]), base):
        r = load_recording(s, n, base)
        m, y = json.loads(str(r["meta_json"])), r["y"]
        rows.append(dict(rec=rec_name(s, n), subject=s, night=n, n_epochs=len(y), n_trim=int(r["in_trim_window"].sum()),
                         n_trim_scored=int((r["in_trim_window"] & (y >= 0)).sum()),
                         **{k: m[k] for k in ("psg_file", "hyp_file", "psg_sha256", "psg_start", "offset_s", "signal_sec",
                                              "hyp_end_s", "n_hyp_epochs", "n_outside_signal")},
                         n_anomalies=len(m["anomalies"]), outside_counts=json.dumps(m["outside_counts"]),
                         **{f"n_{v}": int((y == int(k)).sum()) for k, v in cfg["labels"]["names"].items()}))
        tables.append(recording_flags(r, cfg, meta))
    man = pd.DataFrame(rows).merge(meta, on=["subject", "night"], how="left")
    assert len(man) == cfg["n_recordings"], f"{len(man)} processed recordings"
    man.to_csv(rep / "recording_manifest.csv", index=False)
    flags = pd.concat(tables, ignore_index=True)
    flags.to_parquet(repo_path(base["paths"]["processed_dir"]) / "epoch_flags.parquet", index=False)
    names = {int(k): v for k, v in cfg["labels"]["names"].items()}
    pd.DataFrame([dict(rk_annotation=k, code=v, aasm=names[v], scored=v >= 0) for k, v in cfg["labels"]["map"].items()]).to_csv(rep / "label_map.csv", index=False)
    flags.groupby(["subject", "label"]).size().unstack(fill_value=0).rename(columns=names).to_csv(rep / "label_counts_by_subject.csv")
    nonflat = flags[~flags.flat_any]  # flat epochs' hashes collide trivially; they are flagged instead
    dups = pd.concat([nonflat[nonflat.duplicated("hash", keep=False)].assign(kind="exact_hash"),
                      flags[flags.adj_corr_fpz > cfg["qc"]["near_dup_corr"]].assign(kind="near_adjacent")])
    dups[["subject", "night", "epoch_idx", "label", "hash", "adj_corr_fpz", "kind"]].to_csv(rep / "duplicates.csv", index=False)
    flow = exclusion_flow(flags, man)
    flow.to_csv(rep / "exclusion_flow.csv", index=False)
    d = load_dataset(man.subject.unique(), channels=tuple(cfg["signal"]["channels"]), base=base)
    assert np.isfinite(d["X"]).all() and (np.abs(d["X"]).max(axis=(1, 2)) > 0).all(), "NaN/Inf or all-zero epochs"
    counts = {STAGE[int(k)]: int(v) for k, v in zip(*np.unique(d["y"], return_counts=True))}
    assert counts == cfg["expected_counts"], f"protocol counts {counts} != published {cfg['expected_counts']}"
    facts = missingness_tables(add_context(flags, man), meta, cfg["missingness"]["hours_bin"], rep)
    facts.update(recordings=len(man), subjects=int(man.subject.nunique()), sha256_verified=len(man), schema_issues=0,
                 exact_duplicates=int((dups.kind == "exact_hash").sum()), near_duplicates=int((dups.kind == "near_adjacent").sum()),
                 protocol_counts=counts, protocol_epochs=int(sum(counts.values())))
    (rep / "data_checks.json").write_text(json.dumps(facts, indent=2))
    print(flow.to_string(index=False))


if __name__ == "__main__":
    run()
