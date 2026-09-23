# Evaluation

All results are in [summary.csv](summary.csv): one row per experiment × model, with 95% CIs from a bootstrap
over subjects. The columns are described in [`data/DATA_DICTIONARY.md`](../data/DATA_DICTIONARY.md) section 6.

## Protocol

- **Data:** Sleep-EDF-20, EEG Fpz-Cz, the 42,308 epochs of the AttnSleep / DeepSleepNet protocol
  ([data/README.md](../data/README.md)).
- **Subject-wise 20-fold CV:** leave one subject out, with the same folds for every model
  (`data/splits/sleepedf20_folds.json`). No subject is ever in both training and test, and
  `tests/test_data.py` asserts this. In each fold, 2 of the 19 training subjects are held out as
  validation subjects. They are used only for early stopping and checkpoint choice.
- **Metrics:** accuracy, macro-F1, Cohen's kappa and per-class F1, pooled over all test folds (as the papers
  report them).
- **Uncertainty:** 95% CIs come from 1,000 bootstrap resamples of subjects (not epochs). Models trained with
  several seeds are reported as the seed mean, with their seed range.
- **No test-fold tuning:** epochs, checkpoints and early stopping are chosen on the validation subjects only.
  Normalisation is per recording and uses no labels, so nothing is fitted on test subjects.
- **Deep runs are imported, not retrained:** the deep models were trained once with the exact settings in
  `configs/exp1_main.yaml`. `make exp1` imports those runs (`src/deep/import_runs.py`). On a fresh clone the
  saved runs are not there (they are not in git), and `make exp1` trains them with the same config instead
  (GPU, about 1-2 h per model and seed). The import:
  - checks the saved config against the experiment config, key by key;
  - checks every protocol epoch and fold;
  - records SHA256 provenance.

  `make verify-imports` reloads every checkpoint with the current code and re-predicts
  each test subject. For all 4 runs (AttnSleep, TinySleepNet × 3 seeds) the predictions are identical
  (maximum probability difference 0.0, `results/exp1_main/*/seed*/verify.json`).
  The training code was checked the same way: retraining fold 0 of AttnSleep and TinySleepNet (seed 42) with the
  current code gives the same epochs, the same training losses and the same predictions as the saved runs.

## Model selection: what was fixed in advance and what was not

- **Fixed in advance:**
  - LightGBM uses YASA's published hyperparameters and class weights;
  - AttnSleep and TinySleepNet use their repositories' settings;
  - every model uses the same folds.
- **Chosen after seeing test results: `seqctx_transformer`.** It is our own design (TinySleepNet's CNN plus a
  2-layer Transformer over 20 epochs). It was picked as "our best deep model" after the test-fold results of
  several deep variants were known, which is selection bias. Other variants were tried and are not in this
  repository. To limit that bias:
  - it is reported as the **mean of its 3 training seeds**, not its best seed (seed 42 is also its best:
    84.3% vs 82.5% and 83.4%, `summary.csv` seed columns);
  - `summary.csv` labels it in the `note` column.

  A clean test would fix it as it is and evaluate it once on new subjects, for example those in Sleep-EDF-78
  but not in Sleep-EDF-20.

## Why our AttnSleep reproduction scores below the paper

Our AttnSleep scores 81.4% accuracy (`summary.csv`); the paper reports 84.4% (Eldele et al. 2021, IEEE TNSRE,
Table V). The gap has several known causes:

1. **The paper picks its reported epoch on the test fold.** The repository keeps the epoch with the best
   *test* accuracy (`trainer.py`, `selected_d`). Our run also scored the test subject after every epoch, for
   this audit only, never for selection. Choosing the epoch the paper's way raises our pooled accuracy from
   81.4% to 83.1%, which is **+1.8 points**
   ([attnsleep_selection_audit.json](attnsleep_selection_audit.json); per fold:
   [attnsleep_selection_audit.csv](attnsleep_selection_audit.csv)). This is a lower bound, because:
2. **We train for fewer epochs.** We stop early after at most 40 epochs (the longest fold ran 30). The repo
   trains 100 epochs and then picks the best one on test, so it has more epochs to pick from. In 7 of our
   20 folds, the epoch chosen on validation came before the learning-rate step at epoch 10.
3. **We hold out validation subjects.** We train on 17 subjects per fold instead of 19.
4. **The repo's class weights use test labels.** It counts train + test labels; we count training labels only.
5. **The paper's hyperparameters were chosen "across these folds"** (paper p. 815), which is also test-fold
   tuning.

After item 1, about 1.3 points remain. That is inside our CI (78.7-83.8%) and comparable to seed-to-seed
variation, and we did not separate it further.
