# Sleep staging from single-channel EEG (Sleep-EDF-20)

Under honest subject-wise evaluation, how close does a simple tabular model (LightGBM on EEG features) get to
published deep models for sleep staging, and how much do leaky splits inflate the results? This repository
answers that on Sleep-EDF-20 (20 subjects, 39 nights, EEG Fpz-Cz). It uses the AttnSleep protocol, with
95% CIs from a bootstrap over subjects, and a documented data-quality pipeline: every exclusion is counted, and
every missing-data mechanism is stated with its evidence.

## Headline results (subject-wise 20-fold CV, 42,308 epochs)

<!-- BEGIN:main -->
| Model | Acc % | Macro-F1 % | Kappa | F1 W | F1 N1 | F1 N2 | F1 N3 | F1 REM | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM, YASA-style features + neighbouring-epoch context | 82.4 [79.7, 84.9] | 76.4 [73.0, 79.3] | 0.760 [0.723, 0.794] | 85.9 | 41.7 | 87.2 | 87.9 | 79.3 | 1 seed |
| AttnSleep reproduction (epoch chosen on validation subjects) | 81.4 [78.7, 83.8] | 75.8 [73.0, 78.2] | 0.747 [0.711, 0.780] | 86.1 | 40.6 | 87.3 | 88.7 | 76.1 | 1 seed |
| TinySleepNet (CNN + LSTM, 20-epoch windows) | 82.0 [77.6, 85.4] | 77.2 [72.5, 80.7] | 0.758 [0.699, 0.802] | 86.2 | 47.7 | 85.6 | 82.9 | 83.4 | mean of 3 seeds (acc range 81.8-82.5) |
| seqctx_transformer (TinySleepNet CNN + Transformer; own design) | 83.4 [79.6, 86.4] | 78.7 [74.8, 81.8] | 0.776 [0.724, 0.816] | 87.5 | 50.5 | 86.3 | 83.3 | 85.7 | mean of 3 seeds; chosen as our best deep model after seeing test results (selection bias) (acc range 82.5-84.3) |
| *AttnSleep, published (Eldele et al. 2021, Table V)*¹ | 84.4 | 78.1 | 0.79 | 89.7 | 42.6 | 88.8 | 90.2 | 79.0 | n/a |

Is the published value inside our 95% CI? (inside = "matches")

- lgbm: acc inside, macro-F1 inside, kappa inside
- attnsleep: acc above, macro-F1 inside, kappa above
- tinysleepnet: acc inside, macro-F1 inside, kappa inside
- seqctx_transformer: acc inside, macro-F1 inside, kappa inside
<!-- END:main -->

¹ AttnSleep's published numbers use the epoch with the best accuracy on the **test** fold. Our AttnSleep run
gains 1.8 points from that rule alone ([evaluation/README.md](evaluation/README.md#why-our-attnsleep-reproduction-scores-below-the-paper)).

- **LightGBM matches the published AttnSleep.** The published accuracy, macro-F1 and kappa all lie inside
  LightGBM's 95% CI.
- **Leaky splits inflate the results a lot.** A random epoch-wise split adds about 10 points of accuracy and
  27 points of N1 F1 to the same model ([leakage](evaluation/README.md#leakage-exp2)).
- **seqctx_transformer was chosen after seeing test results**, so it is reported as the mean of 3 seeds. Its
  CI overlaps LightGBM's.

| | |
|---|---|
| Data: source, license, download, processing, exclusion flow | [data/README.md](data/README.md) |
| File formats and every column | [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md) |
| Cleaning decisions, missingness map and mechanisms | [data/data_quality.md](data/data_quality.md) |
| Protocol, all results, leakage, context, missingness, reproduction gap, limitations | [evaluation/README.md](evaluation/README.md) |
| Every number (one row per experiment × model) | [evaluation/summary.csv](evaluation/summary.csv) |

## Quickstart

```bash
uv sync           # Python 3.12 + dependencies (PyTorch from the CUDA 12.8 index)
make data         # download 39 recordings (1.96 GB, SHA256-verified) -> 30-s epochs + data-quality tables
make splits       # subject-wise 20-fold CV -> data/splits/sleepedf20_folds.json
make exp1         # main result: LightGBM (CPU, ~1 min) + deep models (imported, or trained on a GPU)
make exp2 exp3 exp4   # leakage, temporal context, missingness (LightGBM, CPU, a few minutes)
make report       # runs analysis.ipynb: the tables in this README and evaluation/figures/
make test
```

## Layout

```
configs/        base.yaml (data, folds, features, metrics, seed, published values) + one file per experiment
data/           data docs, download script, data-quality tables (reports/), folds (splits/)
src/            prepare_data.py (EDF -> epochs -> checks), splits.py, evaluate.py (metrics, bootstrap CIs),
                runs.py, experiments.py (exp1-exp4), config.py
src/classical/  features.py (YASA-style Fpz-Cz features), train.py (LightGBM CV)
src/deep/       models.py (AttnSleep, TinySleepNet, seqctx_transformer), train.py (deep CV),
                import_runs.py (imports the saved deep runs, with config checks, SHA256 provenance, --verify)
evaluation/     README.md (results and how to read them), summary.csv, figures/
analysis.ipynb  builds the README tables and the figure from evaluation/summary.csv
tests/
```
