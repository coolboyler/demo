# Tianlang Files

This folder is trimmed to the current Tianlang production path.

- `raw/`: source company-detail Excel files used to rebuild the site history.
- `actual/`: monthly actual Excel exports kept as source input.
- `new/`: retained datasets for the current model chain.
  - `history_daily.csv`, `history_hourly.csv`: operational history used by the app and future forecasting.
  - `history_daily_2024fill.csv`, `history_hourly_2024fill.csv`: retained backfilled history used to build the best dataset.
  - `baseline_d6_dataset_2024fill.csv`: current best-model dataset.
  - `dataset_2024fill_summary.json`: summary for the retained best dataset.
- `results/`: only the `2024fill_shared` best-model artifacts and current Tianlang forecast outputs.
  - future forecast files are organized under `results/YYYY-MM/`.
- `train_baseline_d6_tianlang.py`, `train_best_d6_tianlang.py`: defaults aligned to the retained `2024fill_shared` artifacts.
- `build_2024fill_dataset_tianlang.py`: rebuilds the retained 2024 backfilled dataset from operational history.
