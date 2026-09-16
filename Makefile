# One command per step. Each experiment writes results/<experiment>/ and its rows of evaluation/summary.csv.
.PHONY: download data splits features exp1 verify-imports test

download:  ## 39 Sleep-EDF-20 recordings (1.96 GB), SHA256-verified, into data/raw/
	bash data/download_sleepedf20.sh

data: download  ## raw EDF -> data/processed/*.npz + cleaning and missingness tables in data/reports/
	uv run python -m src.prepare_data

splits:  ## subject-wise 20-fold CV -> data/splits/sleepedf20_folds.json
	uv run python -m src.splits

features:  ## YASA-style Fpz-Cz features -> data/processed/features_fpz.parquet
	uv run python -m src.classical.features

exp1:  ## main result: subject-wise 20-fold CV of every model in configs/exp1_main.yaml (deep runs are imported)
	uv run python -m src.experiments exp1

verify-imports:  ## re-predict every imported deep run from its checkpoints with the current code (GPU)
	uv run python -m src.deep.import_runs --experiment exp1_main --verify

test:
	uv run pytest -q
