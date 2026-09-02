# Sleep staging from single-channel EEG (Sleep-EDF-20)

**Question.** Under honest subject-wise evaluation, how close does a simple tabular model (LightGBM on EEG
features) get to published deep models for sleep staging, and how much do leaky splits inflate the results?

This repository builds the evaluation from the data up. It starts with a documented, reproducible data
pipeline: every exclusion is counted, and every missing-data mechanism is stated together with its evidence.
The models and results are added on top of that pipeline.

| | |
|---|---|
| Data source, download, processing, exclusion flow | [data/README.md](data/README.md) |
| File formats and every column | [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md) |
| Cleaning decisions, missingness map and mechanisms | [data/data_quality.md](data/data_quality.md) |

## Quickstart

```bash
uv sync        # Python 3.12 + dependencies (PyTorch from the CUDA 12.8 index)
make data      # download 39 recordings (1.96 GB, SHA256-verified) -> processed epochs + data-quality tables
make splits    # subject-wise 20-fold CV -> data/splits/sleepedf20_folds.json
make test
```

## Layout

```
configs/base.yaml        data paths, preprocessing, QC thresholds, folds, seed
data/                    data docs, download script, data-quality tables (reports/), folds (splits/)
src/prepare_data.py      raw EDF -> 30-s epochs -> cleaning checks -> missingness diagnostics
src/splits.py            subject-wise folds (and the leaky epoch-wise split used to measure leakage)
tests/
```
