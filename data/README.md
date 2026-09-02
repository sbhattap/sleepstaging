# Data: Sleep-EDF-20

## Source, citation and license

- **Dataset:** Sleep-EDF Database Expanded, version 1.0.0, on PhysioNet:
  <https://physionet.org/content/sleep-edfx/1.0.0/> (DOI [10.13026/C2X676](https://doi.org/10.13026/C2X676)).
- **License:** Open Data Commons Attribution License v1.0 (ODC-By). The raw data and the processed arrays are
  not redistributed in this repository; the scripts below rebuild them from PhysioNet.
- **Cite:**
  - B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback
    loop: the slow-wave microcontinuity of the EEG. *IEEE Trans. Biomed. Eng.* 47(9):1185-1194 (2000).
  - AL Goldberger, LAN Amaral, L Glass, JM Hausdorff, PCh Ivanov, RG Mark, JE Mietus, GB Moody, C-K Peng,
    HE Stanley. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex
    physiologic signals. *Circulation* 101(23):e215-e220 (2000).

## What we use

"Sleep-EDF-20" is the subset used by DeepSleepNet (Supratak et al. 2017) and AttnSleep (Eldele et al. 2021):

- **Subjects:** Sleep Cassette (SC) subjects 0-19, which is 20 healthy adults aged 25-34, 10 female and
  10 male (`data/reports/subject_metadata.csv`).
- **Recordings:** 39 nights. Subject 13 has only night 1; PhysioNet says that night was "lost due to a
  failing cassette or laserdisk".
- **Signals:** each night is about 21-24 h of EEG (Fpz-Cz, Pz-Oz), horizontal EOG, submental EMG and other
  channels.
- **Labels:** a hypnogram of 30-s epochs scored by technicians with the Rechtschaffen & Kales (R&K) rules.

File formats, channels and every derived file: [DATA_DICTIONARY.md](DATA_DICTIONARY.md).
Cleaning decisions and missing data: [data_quality.md](data_quality.md).

## How to get the data

AttnSleep's download script ([gist](https://gist.github.com/emadeldeen24/a22691e36759934e53984289a94cb09b))
fetches the same 39 recordings with `wget` from PhysioNet's retired `physiobank` path. Our version,
[`download_sleepedf20.sh`](download_sleepedf20.sh), does the same with the current URL and some safeguards:

- **Current URL:** it downloads from <https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/>.
- **File names from PhysioNet's list:** names are read from `SHA256SUMS.txt`, because hypnogram names do not
  follow PSG names. The 8th character identifies the scoring technician: `SC4001E0-PSG.edf` goes with
  `SC4001EC-Hypnogram.edf`, and `SC4011E0-PSG.edf` with `SC4011EH-Hypnogram.edf`.
- **Checksums:** every file is verified against `SHA256SUMS.txt`. The script stops on a mismatch, on any
  night that lacks exactly one PSG and one hypnogram, or if the total is not 39 recordings.
- **Safe to re-run:** files that are already downloaded and verified are skipped.

```bash
bash data/download_sleepedf20.sh               # 78 files (39 PSG + 39 hypnograms) -> data/raw/SC4<ss><n>/
SUBJECTS="0 1" bash data/download_sleepedf20.sh # a subset, e.g. for a quick test
```

**Size and disk space:**

| What | Size |
|---|---|
| Download: 39 PSG files (47-52 MB each) + 39 hypnograms (2-6 kB each) | **1.96 GB** (`data/reports/physionet_files.csv`) |
| Processed `.npz` files (3 channels, whole recording, float32) | about 3.7 GB (39 × about 95 MB) |
| Peak disk during `make data` | about 5.7 GB; each raw recording is deleted once it has been processed |

`prepare_data` re-checks every file's SHA256 before using it and skips recordings that are already processed.

## How it is processed

```bash
uv sync                          # environment (Python 3.12, see pyproject.toml)
make data                        # = bash data/download_sleepedf20.sh + uv run python -m src.prepare_data
make splits                      # = uv run python -m src.splits -> data/splits/sleepedf20_folds.json
```

All steps are in one file, [`src/prepare_data.py`](../src/prepare_data.py), with settings in
[`configs/base.yaml`](../configs/base.yaml) (`data:` section).

### Steps for each recording

1. **Integrity:** check the SHA256 of the PSG and hypnogram against PhysioNet's `SHA256SUMS.txt`. Exactly one
   PSG and one hypnogram per night, 39 nights.
2. **Header validation:**
   - the 7 expected channels are present;
   - EEG and EOG are at 100 Hz in µV, and the EMG envelope is at 1 Hz;
   - the PSG and hypnogram start times are identical (offset 0 s in all 39 recordings).
3. **Channel selection:** keep EEG Fpz-Cz, EEG Pz-Oz and EOG horizontal as signals, plus the per-epoch mean
   of the EMG envelope (used only in the missingness checks). The models use **Fpz-Cz only**. Respiration,
   rectal temperature and the event marker are not used.
4. **No resampling or filtering:** the signals are already at 100 Hz (3,000 samples per 30-s data record),
   and DeepSleepNet and AttnSleep also feed raw µV.
5. **Epoching:** the hypnogram's annotations (onset, duration, label) are expanded to a 30-s grid that starts
   at the signal start. Epochs past the end of the signal are dropped: every hypnogram covers 24 h, but the
   recorder stopped earlier, and all of those epochs are '?'. Each kept epoch is a `[3, 3000]` slice.
6. **Label mapping (R&K → AASM):** W→W, 1→N1, 2→N2, **3 and 4→N3**, R→REM. Movement time → −1 and
   '?' → −2 are kept in the file, so they can be counted, but are never scored.
7. **Wake trim mask:** `in_trim_window` marks the AttnSleep / DeepSleepNet window. Movement and '?' epochs are
   removed first, then the window runs from 30 min (60 epochs) before the first non-wake epoch to 30 min
   after the last. The mask is stored, not applied, so the untrimmed night stays available.
8. **Save:** the result goes to `data/processed/SC4<ss><n>.npz`, and the raw EDFs are then deleted.

### Checks on the whole dataset

- schema, units, labels and group IDs for every recording (any problem stops the run);
- exact and near duplicates (none found; `data/reports/duplicates.csv` is empty);
- artifact flags for flat, clipped and high-amplitude epochs, stored as columns in
  `data/processed/epoch_flags.parquet`, not deleted;
- the exclusion flow;
- post-cleaning checks: no NaN or Inf, no empty epochs, only valid labels, and **the final per-class counts must
  equal AttnSleep's Table I**, or the run fails;
- missingness diagnostics.

Every table is written to [`reports/`](reports/), and the scalar results of the checks to
[`reports/data_checks.json`](reports/data_checks.json).

**At load time** (`src.prepare_data.load_dataset`), the protocol keeps the epochs inside the wake-trim window
and drops Movement and '?' epochs. This is exactly the AttnSleep epoch set.

## Exclusion flow (Sleep-EDF-20)

Copied from [`reports/exclusion_flow.csv`](reports/exclusion_flow.csv):

| Step | Removed | Remaining | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| Hypnogram epochs, all 39 recordings | 0 | 112,320 | | | | | |
| Drop epochs past the end of the signal (truncated tail, all '?') | 4,940 | 107,380 | 72,391 | 2,804 | 17,799 | 5,703 | 7,717 |
| Drop Unknown '?' epochs inside the signal | 905 | 106,475 | 72,391 | 2,804 | 17,799 | 5,703 | 7,717 |
| Drop Movement time epochs | 61 | 106,414 | 72,391 | 2,804 | 17,799 | 5,703 | 7,717 |
| Drop wake more than 30 min before or after sleep (AttnSleep / DeepSleepNet trim) | 64,106 | **42,308** | 8,285 | 2,804 | 17,799 | 5,703 | 7,717 |
| Artifact-flagged Fpz-Cz epochs, kept (flat 0, clipped 27, high amplitude 176; 195 epochs with any flag) | 0 | 42,308 | | | | | |

The final per-class counts equal AttnSleep's Table I exactly.
