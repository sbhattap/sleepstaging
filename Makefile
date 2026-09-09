# One command per step. Each experiment writes results/<experiment>/ and its rows of evaluation/summary.csv.
.PHONY: download data splits features exp1 test

download:  ## 39 Sleep-EDF-20 recordings (1.96 GB), SHA256-verified, into data/raw/
	bash data/download_sleepedf20.sh

data: download  ## raw EDF -> data/processed/*.npz + cleaning and missingness tables in data/reports/
	uv run python -m src.prepare_data

splits:  ## subject-wise 20-fold CV -> data/splits/sleepedf20_folds.json
	uv run python -m src.splits

features:  ## YASA-style Fpz-Cz features -> data/processed/features_fpz.parquet
	uv run python -m src.classical.features

exp1:  ## main result: subject-wise 20-fold CV of every model in configs/exp1_main.yaml
	uv run python -m src.experiments exp1

test:
	uv run pytest -q
