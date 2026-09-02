# Data dictionary

Every entry was checked against the files themselves:
- the EDF headers of `SC4001E0-PSG.edf` and `SC4001EC-Hypnogram.edf`, downloaded and SHA256-verified;
- the header fields of all 39 Sleep-EDF-20 recordings, saved in each processed file's `meta_json`;
- the processed `.npz` arrays;
- the derived tables in `data/reports/`.

Ranges marked "39 recordings" were taken across all of them.

## 1. Raw files (PhysioNet `sleep-cassette/`)

### File names

- **PSG:** `SC4<ss><N>E0-PSG.edf`, where `ss` = subject (00-19 here) and `N` = night (1 or 2).
- **Hypnogram:** `SC4<ss><N>E<x>-Hypnogram.edf`. `<x>` is a letter that identifies the scoring technician
  (PhysioNet: "identified by the eighth letter of the hypnogram filename"), for example `SC4001EC`,
  `SC4011EH` and `SC4192EV`. It must be read from PhysioNet's file list; it cannot be derived from the PSG name.
- **Missing:** `SC4132*` does not exist, because night 2 of subject 13 was lost.

### PSG `.edf`: EDF, one 30-s data record, 7 signals

| Signal label | Transducer | Unit | Samples per 30-s record → rate | Physical range | Digital range | Prefilter (header) | Used |
|---|---|---|---|---|---|---|---|
| `EEG Fpz-Cz` | Ag-AgCl electrodes | uV | 3000 → 100 Hz | about ±176 to ±209 (39 recordings) | −2048 to 2047 | HP 0.5 Hz, LP 100 Hz | **model input** |
| `EEG Pz-Oz` | Ag-AgCl electrodes | uV | 3000 → 100 Hz | about ±173 to ±212 (39 recordings) | −2048 to 2047 | HP 0.5 Hz, LP 100 Hz | stored, QC only |
| `EOG horizontal` | Ag-AgCl electrodes | uV | 3000 → 100 Hz | about ±850 to ±1032 (39 recordings) | −2048 to 2047 | HP 0.5 Hz, LP 100 Hz | stored, QC only |
| `Resp oro-nasal` | oral-nasal thermistors | (none) | 30 → 1 Hz | −2048 to 2047 | −2048 to 2047 | HP 0.03 Hz, LP 0.9 Hz | no |
| `EMG submental` | Ag-AgCl electrodes | uV | 30 → 1 Hz | −5 to 5 | −2500 to 2500 | HP 16 Hz, rectification, LP 0.7 Hz (an envelope, not raw EMG) | per-epoch mean, missingness checks only |
| `Temp rectal` | rectal thermistor | DegC | 30 → 1 Hz | 34 to 40 | −2849 to 2731 | (none) | no |
| `Event marker` | marker button | (none) | 30 → 1 Hz | −2047 to 2048 | −2047 to 2048 | "Hold during 2 seconds" | no |

Notes:
- The digital range and prefilter columns come from the `SC4001E0` header.
- All 39 recordings have the same 7 signal labels in the same order, and the same units and rates for the
  4 stored channels.
- Recording length is 21.4-23.9 h (39 recordings).
- The patient field is anonymised to sex and age, e.g. `X F X Female_33yr`.

### Hypnogram `.edf`: EDF+C with a single `EDF Annotations` signal

Each annotation is an (onset in s, duration in s, label) triple that covers a run of epochs:
- The first annotation starts at 0 s.
- Every duration is a multiple of 30 s (0 exceptions in 39 recordings).
- The hypnogram start time equals the PSG start time in all 39 recordings.
- Every hypnogram covers 86,400 s (24 h), which is longer than the signal.

| Annotation label | Meaning | 30-s epochs inside the signal (39 recordings) |
|---|---|---|
| `Sleep stage W` | wake | 72,391 |
| `Sleep stage 1` | R&K stage 1 | 2,804 |
| `Sleep stage 2` | R&K stage 2 | 17,799 |
| `Sleep stage 3` | R&K stage 3 | 3,370 |
| `Sleep stage 4` | R&K stage 4 | 2,333 |
| `Sleep stage R` | REM | 7,717 |
| `Movement time` | movement, not scored | 61 |
| `Sleep stage ?` | not scored | 905 (plus 4,940 past the end of the signal) |

### Subject metadata

The source is PhysioNet's `SC-subjects.xls`, read through MNE's copy
(`mne/datasets/sleep_physionet/age_records.csv`). Our extract is `data/reports/subject_metadata.csv`:

| Column | Type | Meaning |
|---|---|---|
| `subject` | int | SC subject number (0-19 here) |
| `night` | int | 1 or 2 |
| `age` | int, years | 25-34 in Sleep-EDF-20 |
| `sex` | str | `female` or `male` (10 / 10 subjects) |
| `lights_off` | str, `HH:MM:SS` | clock time of lights off for that night |

## 2. Label mapping (R&K → AASM)

From `configs/base.yaml` `data.labels.map`; also written to `data/reports/label_map.csv`.

| R&K annotation | Code | AASM class | Scored |
|---|---|---|---|
| Sleep stage W | 0 | W | yes |
| Sleep stage 1 | 1 | N1 | yes |
| Sleep stage 2 | 2 | N2 | yes |
| Sleep stage 3 | 3 | N3 | yes |
| Sleep stage 4 | 3 | N3 (merged with stage 3) | yes |
| Sleep stage R | 4 | REM | yes |
| Movement time | −1 | MOVE | no: excluded from training and scoring |
| Sleep stage ? | −2 | UNK | no: excluded from training and scoring |

## 3. Processed data: `data/processed/SC4<ss><N>.npz` (one per recording, untrimmed)

`n` is the number of 30-s epochs inside the signal: 2,570-2,864 per recording.

| Array | Shape | Dtype | Unit / values |
|---|---|---|---|
| `x` | `[n, 3, 3000]` | float32 | µV; channels in the order of `ch_names`; 100 Hz × 30 s; no filtering or resampling |
| `ch_names` | `[3]` | str | `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal` (models use index 0, Fpz-Cz) |
| `y` | `[n]` | int8 | 0 W, 1 N1, 2 N2, 3 N3, 4 REM, −1 Movement, −2 '?' |
| `y_raw` | `[n]` | str | original R&K annotation label |
| `subject`, `night` | scalar | int64 | IDs; group key for the subject-wise splits |
| `sfreq` | scalar | float64 | 100.0 Hz |
| `epoch_onset_s` | `[n]` | float64 | s from recording start (0, 30, 60, ...) |
| `in_trim_window` | `[n]` | bool | inside the AttnSleep / DeepSleepNet window (±30 min of wake around sleep) |
| `emg_epoch_mean` | `[n]` | float32 | mean of the 1-Hz EMG envelope over the epoch, µV |
| `phys_min`, `phys_max` | `[3]` | float64 | EDF header physical range per stored channel, µV (used for clipping detection) |
| `meta_json` | scalar | str (JSON) | file names, SHA256, start times, offset, signal length, epochs past the signal end, header labels, per-channel rate, unit and range |

Protocol epochs come from `src.prepare_data.load_dataset(trim=True, drop_unscored=True)`, which keeps
`in_trim_window & (y >= 0)`: 42,308 epochs for Sleep-EDF-20.

## 4. Derived tables (`src/prepare_data.py`)

### `data/processed/epoch_flags.parquet`: one row per epoch of every recording (git-ignored, rebuilt by `make data`)

| Column | Unit | Definition |
|---|---|---|
| `subject`, `night`, `epoch_idx` | | epoch key (`epoch_idx` = position in the `.npz`) |
| `label`, `label_raw` | | as `y` and `y_raw` above |
| `onset_s` | s | epoch onset |
| `in_trim_window` | bool | as above |
| `emg_mean` | µV | as `emg_epoch_mean` |
| `sd_<ch>`, `ptp_<ch>`, `max_abs_<ch>` | µV | standard deviation, peak-to-peak and maximum absolute value of the epoch |
| `flat_run_s_<ch>` | s | longest run of identical consecutive samples |
| `clip_frac_<ch>` | 0-1 | fraction of samples within 0.2 % of the header range of either rail |
| `log_sd_z_<ch>` | z | robust z-score of log(SD), with median and MAD taken over that recording's scored epochs inside the trim window |
| `flat_<ch>` | bool | SD < 1 µV, or a flat run ≥ 5 s |
| `clipped_<ch>` | bool | `clip_frac` ≥ 1 % |
| `high_amp_<ch>` | bool | `log_sd_z` > 3 |
| `flat`, `clipped`, `high_amp` | bool | the Fpz-Cz flags (the modelling channel) |
| `flat_any`, `clipped_any`, `high_amp_any` | bool | the flag on any of the 3 stored channels |
| `adj_corr_fpz` | r | Pearson correlation with the previous epoch (near-duplicate check, threshold 0.99) |
| `hash` | str | hash of the epoch's bytes over all 3 channels (exact-duplicate check) |

`<ch>` is one of `fpz`, `pz`, `eog`. The thresholds are in `configs/base.yaml` `data.qc`.

### Tables in `data/reports/` (tracked)

| File | Columns (summary) |
|---|---|
| `recording_manifest.csv` | one row per recording: `rec`, `subject`, `night`, `n_epochs`, `n_trim`, `n_trim_scored`, file names, PSG SHA256, `psg_start`, `offset_s`, `signal_sec`, `hyp_end_s`, `n_hyp_epochs`, `n_outside_signal`, `n_anomalies`, `outside_counts`, per-class counts `n_W` … `n_UNK`, `age`, `sex`, `lights_off` |
| `physionet_files.csv` | `subject`, `night`, `psg_file`, `hyp_file`, `psg_bytes`, `hyp_bytes`, `psg_sha256`, `hyp_sha256` |
| `subject_metadata.csv` | section 1, Subject metadata |
| `exclusion_flow.csv` | `step`, `removed`, `remaining`, `note`, per-class counts `W` … `REM` after the step |
| `label_map.csv` | `rk_annotation`, `code`, `aasm`, `scored` (section 2) |
| `label_counts_by_subject.csv` | `subject` × class (all epochs inside the signal, incl. MOVE and UNK) |
| `duplicates.csv` | `subject`, `night`, `epoch_idx`, `label`, `hash`, `adj_corr_fpz`, `kind` (`exact_hash` / `near_adjacent`); empty for Sleep-EDF-20 |
| `missingness_nights_per_subject.csv` | `subject`, `n_nights`, `age`, `sex` (whole SC release, metadata only) |
| `missingness_unscored_by_subject.csv` | `subject`, `is_move`, `is_unk`, `n`, `rate_pct` (inside the trim window) |
| `missingness_next_unscored_by_stage.csv` | stage, `sum`, `count`, `rate_pct`: P(next epoch unscored given the current stage) |
| `missingness_logit_unscored.csv`, `missingness_logit_unknown_fullnight.csv`, `artifact_logit.csv` | `outcome`, `term`, `lr_chi2`, `df`, `p_value`: likelihood-ratio test for dropping each term from a logistic model |
| `artifact_rate_by_stage.csv`, `artifact_count_by_stage.csv` | flag (`flat`, `clipped`, `high_amp`, `artifact` = any Fpz-Cz flag, `*_any`) × stage: % or count of protocol epochs |
| `data_checks.json` | scalar results: recordings, subjects, duplicates, protocol counts, Mann-Whitney / Fisher p-values, EMG medians |

### Folds: `data/splits/sleepedf20_folds.json` (tracked)

A list of 20 folds: `fold`, `test_subjects` (1 each), `val_subjects` (2), `train_subjects` (17), `dataset`,
`seed`, `protocol`. It is regenerated identically by `make splits`, and `tests/test_data.py` checks this.
