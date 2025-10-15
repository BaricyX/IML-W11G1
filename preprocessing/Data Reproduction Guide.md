# Data Reproduction Guide

**Input**

- `GUIDE_Test.csv`

**Script**

- `dataset.py`

**Command**

```python
python dataset.py \
  --in GUIDE_Test.csv \
  --out-sample sample_100000.csv \
  --out-train train.csv \
  --out-test test.csv \
  --sample-size 10000 \
  --train-size 80000 \
  --test-size 20000 \
  --seed 2025 \

```

**What it does**

1. Stratified sampling (by `IncidentGrade`)
Draws a stratified sample of `--sample-size` rows from the input CSV.
2. Saves the sample
Writes the sampled rows to `--out-sample`.
3. Stratified train/test split
Splits the sample into train/test using absolute counts (`--train-size`, `--test-size`), still stratified by `IncidentGrade`, and saves to `--out-train` and `--out-test`.
4. Reproducibility
`--seed` controls randomness. Using the same seed reproduces identical sample and split; changing it yields a different but similarly distributed result.

**Outputs**

- `sample_100000.csv`: stratified 100,000-row sample
- `train.csv`, `test.csv`: 80/20 stratified splits of the sample

**Reproduce**

- Using the same command and `--seed 2025` yields identical results.
- Changing `--seed` produces a different but similarly distributed sample.

