import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_sample(df: pd.DataFrame, by: str, n_total: int, seed: int) -> pd.DataFrame:
    # Return a deterministic stratified sample matching the class distribution in column `by` given the seed.
    rng = np.random.default_rng(seed)

    # group by label
    groups = df.groupby(by, dropna=False)
    counts = groups.size()
    total = counts.sum()

    # initial allocation
    proportions = counts / total
    target_per_class = (proportions * n_total).round().astype(int)

    # sample inside each class
    parts = []
    for k, sub_df in groups:
        want = target_per_class.get(k, 0)
        want = max(want, 0)
        want = min(want, len(sub_df))
        if want > 0:
            # per-class seed for reproducibility
            sub_seed = int(rng.integers(0, 2**32 - 1))
            parts.append(sub_df.sample(n=want, random_state=sub_seed))

    # concat and final shuffle
    out = pd.concat(parts, axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return out

def main():
    # sample → save → split train/test (stratified by IncidentGrade) → save
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", default="GUIDE_Test.csv")
    ap.add_argument("--out-sample", default="sample_100000.csv")
    ap.add_argument("--out-train",  default="train.csv")
    ap.add_argument("--out-test",   default="test.csv")
    ap.add_argument("--sample-size", type=int, default=100000)
    ap.add_argument("--train-size",  type=int, default=80000)
    ap.add_argument("--test-size",   type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    label = "IncidentGrade"
    df = pd.read_csv(args.input_csv, low_memory=False)
    n_total = min(args.sample_size, len(df))

    sample_df = stratified_sample(df, by=label, n_total=n_total, seed=args.seed)
    sample_df.to_csv(args.out_sample, index=False)

    train_df, test_df = train_test_split(
        sample_df,
        train_size=args.train_size,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=sample_df[label]
    )
    train_df.to_csv(args.out_train, index=False)
    test_df.to_csv(args.out_test, index=False)

if __name__ == "__main__":
    main()