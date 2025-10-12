# Data Reproduction Guide

**Input**

- `GUIDE_Train.csv`

**Script**

- `dataset.py` (cleaning → stratified sampling → 80/20 train/test split)

**Command**

```python
python dataset.py \
  --in GUIDE_Train.csv \
  --sample-size 10000 \
  --seed 2025 \
  --out-clean cleaned.csv \
  --out-sample sample_10000.csv \
  --out-train train.csv \
  --out-test test.csv
```

**What it does**

1. Normalizes column names, trims strings, parses timestamps, removes duplicates, filters rows with excessive missing values, and applies IQR clipping on numeric columns.
2. Automatically chooses a label column (priority: `LastVerdict` → `SuspicionLevel` → `Category` → `EntityType`) and performs stratified sampling to produce 10,000 rows.
3. Creates an 80/20 stratified train/test split from the sample (controlled by `--seed` for reproducibility).

**Outputs**

- `cleaned.csv`: cleaned full dataset
- `sample_10000.csv`: 10,000-row stratified sample
- `train.csv`, `test.csv`: 80/20 stratified splits of the sample

**Reproduce**

- Using the same command and `--seed 2025` yields identical results.
- Changing `--seed` produces a different but similarly distributed sample.

**Command to reproduce**

```python
python dataset.py --in GUIDE_Train.csv --sample-size 10000 --seed 2025 --out-clean cleaned.csv --out-sample sample_10000.csv --out-train train.csv --out-test test.csv
```

