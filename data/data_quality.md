# Data quality: cleaning, missing data and their handling

No model is trained before these checks. Each rule is a function in `src/prepare_data.py`, tested in
`tests/test_data.py`. All results are written to [reports/](reports/), and the scalar ones to
[reports/data_checks.json](reports/data_checks.json). Numbers below are copied from the files named next to them.

## 1. Cleaning decisions

| # | Rule | Result (Sleep-EDF-20) | Decision |
|---|---|---|---|
| 1 | Integrity: SHA256 vs PhysioNet; one PSG + one hypnogram per night | 39/39 recordings pass; 1 night (subject 13, night 2) absent in the release | stop on any mismatch |
| 2 | Schema and units: channel labels, 100 Hz EEG/EOG, µV, record length = signal length | all 39 identical (`DATA_DICTIONARY.md` section 1) | stop on anything unexpected; nothing is coerced |
| 3 | Label validity: only the 8 known R&K labels; map to AASM | 8 labels found, mapped as in `reports/label_map.csv` | N3 + N4 → N3 (AttnSleep / DeepSleepNet) |
| 4 | Alignment: hypnogram vs PSG start, 30-s grid | offset 0 s in 39/39; every annotation duration is a multiple of 30 s | epochs start at the signal start |
| 5 | Duplicates: exact (byte hash) and near (adjacent-epoch r > 0.99) | none (`reports/duplicates.csv`) | nothing removed |
| 6 | Quality flags: flat / clipped / high-amplitude per channel | 195 Fpz-Cz-flagged epochs among the 42,308 protocol epochs | **flag, do not delete**, so their effect can be tested (section 4) |
| 7 | Group IDs: subject and night parsed from the file and matched to the metadata | 39/39 match; 20 subjects | splits use the subject as the group |
| 8 | Exclusion flow | 112,320 → 42,308 epochs (`reports/exclusion_flow.csv`, table in [README.md](README.md)) | the AttnSleep protocol, reproduced exactly |
| 9 | Post-cleaning checks: no NaN or Inf, no all-zero epochs, labels in 0-4, aligned lengths | pass | stop on failure |

**The final count matches the published one.** 42,308 epochs with W/N1/N2/N3/REM = 8,285 / 2,804 / 17,799 /
5,703 / 7,717 is exactly AttnSleep's Table I. We also re-derived the count without our own loader, by
re-running the AttnSleep / DeepSleepNet selection rule on the PhysioNet hypnograms, and it gave the same epoch
set in all 39 recordings. DeepSleepNet's paper reports 41,950 epochs (358 fewer W); that number could not be
reproduced from release 1.0.0.

## 2. Missingness map (Sleep-EDF-20)

| What is missing | Level | How much | Where |
|---|---|---|---|
| Second night of subject 13 | group (night) | 1 of 40 nights | subject 13 |
| Signal shorter than the 24-h hypnogram | recording tail | 4,940 epochs, all '?' | all 39 recordings |
| Unknown '?' epochs inside the signal | label | 905, all **outside** the scored window (leading and trailing blocks, and daytime wake) | `reports/missingness_unscored_by_subject.csv` |
| Movement time epochs | label | 61, all **inside** the scored window (0.14 % of 42,369) | 0-1.02 % per subject |
| Artifact epochs (Fpz-Cz flat / clipped / high amplitude) | signal segment | 195 of 42,308 protocol epochs (0.46 %) | 188 in W, 5 in N3, 2 in N2, 0 in N1 and REM |
| Other modalities | modality | none; all 39 have every channel | |
| Wake more than 30 min from sleep | by design | 64,106 W epochs | not missing data: the protocol removes them |

## 3. Assumed mechanism and evidence

| Missingness | Assumed mechanism | Evidence | Source |
|---|---|---|---|
| Whole night (subject 13) | **MCAR** (administrative) | PhysioNet states the night was lost to "a failing cassette or laserdisk". Across the SC release, single-night subjects do not differ in age (Mann-Whitney p = 0.43) or sex (Fisher p = 0.10), but there are only 3 of them, so the test has little power | `reports/data_checks.json`, `reports/missingness_nights_per_subject.csv` |
| Truncated tail | **MCAR** (by design) | the hypnogram is a fixed 24-h grid and the recorder stopped earlier; every lost epoch is '?', so 0 scored epochs are lost | `reports/recording_manifest.csv` |
| Movement epochs | **MAR** | logistic regression of "unscored" on previous stage, next stage, time of night and subject (likelihood-ratio tests): previous stage p = 6.8e-16, next stage p = 4.2e-12, subject p = 1.2e-16, time of night p = 0.86. Missingness depends on observed variables, so it is not MCAR. MNAR (dependence on the hidden true stage beyond what the neighbours explain) cannot be tested | `reports/missingness_logit_unscored.csv` |
| '?' epochs (whole night) | **MAR** | strongly related to time of night and subject (both p < 1e-300), not to the neighbouring stage; they sit at the edges of the recording and in daytime wake | `reports/missingness_logit_unknown_fullnight.csv` |
| Artifact epochs | **MAR given the stage; MNAR for a deployed model** | rate by stage W 2.27 %, N1 0 %, N2 0.01 %, N3 0.09 %, REM 0 %; stage p = 4.2e-90, time of night p = 4.6e-33, subject p = 7.6e-44; higher EMG in artifact epochs (median 2.95 vs 2.34 µV, Mann-Whitney p = 6.5e-10), which points to body movement. At test time the stage is exactly what we predict, so for a model the missingness depends on the unobserved target | `reports/artifact_rate_by_stage.csv`, `reports/artifact_logit.csv`, `reports/data_checks.json` |

A note on the artifact flags:
- They are computed from the signal: the amplitude, flat runs and samples at the rails of the header range.
- The high-amplitude flag compares each epoch with that recording's scored, in-window epochs. That reference
  set is defined by the labels (the same window the protocol scores), but no epoch's own stage is used.

## 4. Handling choices and sensitivity checks

| Missingness | Main handling | Sensitivity check |
|---|---|---|
| Whole night | keep all available nights; folds are by subject, so subject 13 is simply a one-night test subject | none needed: a single subject, and MCAR |
| Truncated tail | trim to the overlap of signal and labels | none needed |
| Movement / '?' | excluded from training and scoring (standard protocol), counted in the exclusion flow | score Movement epochs as the previous or next stage instead of excluding them |
| Artifacts | keep the epochs; flags are stored as columns | (a) drop flagged epochs; (b) keep them and add the flags as model features; MNAR bounds: assume every flagged epoch is truly W, or truly N1 (the worst case for macro-F1) |
| Wake trim | AttnSleep protocol (±30 min) | documented as a design choice, not missing data |

**Results of these sensitivity checks** (`make exp4`) are in
[evaluation/README.md](../evaluation/README.md#missingness-exp4).

**Preprocessing is never fitted on test subjects.** Normalisation is per recording and uses no labels: the
deep models use the robust median and IQR of that recording's untrimmed night, and the LightGBM features use
per-recording scaling. No imputation model is fitted anywhere, so nothing estimated on one subject is applied
to another.
