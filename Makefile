# One command per step. Later experiments add exp1-exp4 and report targets.
.PHONY: download data splits test

download:  ## 39 Sleep-EDF-20 recordings (1.96 GB), SHA256-verified, into data/raw/
	bash data/download_sleepedf20.sh

data: download  ## raw EDF -> data/processed/*.npz + cleaning and missingness tables in data/reports/
	uv run python -m src.prepare_data

splits:  ## subject-wise 20-fold CV -> data/splits/sleepedf20_folds.json
	uv run python -m src.splits

test:
	uv run pytest -q
