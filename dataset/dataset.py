# -*- coding: utf-8 -*-

import argparse, os, sys, re
import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None

# ------------------ helpers ------------------
def norm_cols(cols: pd.Index) -> pd.Index:
    return (cols.astype(str).str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
            .str.lower())

def trim_obj(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
    return df

def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # parse Timestamp-like columns
    for c in df.columns:
        if c in ("timestamp",) or any(k in c for k in ("time","dt")):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            except Exception:
                pass
    return df

def validate_sha256(s: pd.Series) -> pd.Series:
    # keep valid 64-hex strings; else set NaN (optional)
    pat = re.compile(r"^[a-fA-F0-9]{64}$")
    return s.where(s.astype(str).str.match(pat), other=np.nan)

def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    # prefer Id if present; else full-row dedupe
    if "id" in df.columns:
        return df.drop_duplicates(subset=["id"])
    return df.drop_duplicates()

def drop_rows_too_many_nans(df: pd.DataFrame, nan_ratio=0.8) -> pd.DataFrame:
    thresh = int((1 - nan_ratio) * df.shape[1])
    return df.dropna(thresh=thresh)

def clip_outliers_iqr(df: pd.DataFrame, factor=3.0) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower, upper = q1 - factor*iqr, q3 + factor*iqr
        df[c] = df[c].clip(lower=lower, upper=upper)
    return df

def choose_label(df: pd.DataFrame, user_label: str | None):
    if user_label and user_label in df.columns and df[user_label].nunique(dropna=True) >= 2:
        return user_label
    # auto-detect by priority
    for col in ["lastverdict", "suspicionlevel", "category", "entitytype"]:
        if col in df.columns and df[col].nunique(dropna=True) >= 2:
            return col
    return None

def stratified_draw(df: pd.DataFrame, by: str, n_total: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grp = df.groupby(by, dropna=False)
    sizes = grp.size()
    props = sizes / sizes.sum()
    alloc = (props * n_total).round().astype(int)

    # balance to exact n_total
    diff = n_total - int(alloc.sum())
    if diff != 0:
        for k in sizes.sort_values(ascending=False).index:
            if diff == 0: break
            alloc[k] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1

    parts = []
    for k, sub in grp:
        take = min(len(sub), max(alloc.get(k, 0), 0))
        if take > 0:
            parts.append(sub.sample(n=take, random_state=rng.integers(0, 2**32-1)))
    out = pd.concat(parts, axis=0) if parts else df.head(0)

    if len(out) < n_total and len(df) > len(out):
        need = n_total - len(out)
        leftover = df.drop(index=out.index, errors="ignore")
        if need > 0 and len(leftover) > 0:
            out = pd.concat([
                out,
                leftover.sample(n=min(need, len(leftover)),
                                random_state=np.random.default_rng(seed+1).integers(0, 2**32-1))
            ], axis=0)

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser(description="Clean large CSV, stratified sample 10000, and train/test split.")
    ap.add_argument("--in", dest="input_csv", required=True)
    ap.add_argument("--out-clean", dest="out_clean", default="cleaned.csv")
    ap.add_argument("--out-sample", dest="out_sample", default="sample_10000.csv")
    ap.add_argument("--out-train", dest="out_train", default="train.csv")
    ap.add_argument("--out-test", dest="out_test", default="test.csv")
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--label-col", type=str, default=None, help="Target/stratify column (e.g., LastVerdict)")
    ap.add_argument("--dropna-cols", nargs="*", default=None,
                    help="Required columns; drop rows if any of these is NA (e.g., Id EntityType)")
    args = ap.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Input not found: {args.input_csv}", file=sys.stderr); sys.exit(1)

    # Read
    df = pd.read_csv(args.input_csv, low_memory=False)

    # Normalize columns
    df.columns = norm_cols(pd.Index(df.columns))

    # Light cleaning tailored to your fields
    df = trim_obj(df)
    df = parse_timestamp(df)

    if "sha256" in df.columns:
        df["sha256"] = validate_sha256(df["sha256"])

    # Required field constraint (optional)
    if args.dropna_cols:
        must = [c.lower() for c in args.dropna_cols if c.lower() in df.columns]
        if must:
            before = len(df)
            df = df.dropna(subset=must)
            print(f"[dropna] dropped {before - len(df)} rows due to NA in {must}")

    # Dedupe (prefer Id)
    before = len(df)
    df = dedupe(df)
    print(f"[dedupe] dropped {before - len(df)} duplicate rows")

    # Remove rows with too many NaNs; clip numeric outliers
    before = len(df)
    df = drop_rows_too_many_nans(df, nan_ratio=0.8)
    print(f"[nan-ratio] dropped {before - len(df)} rows with too many NaNs")
    df = clip_outliers_iqr(df, factor=3.0)

    # Save cleaned
    df.to_csv(args.out_clean, index=False)
    print(f"[save] cleaned -> {args.out_clean} (rows={len(df)})")

    # Choose label for stratification
    label = choose_label(df, args.label_col)
    if label:
        print(f"[label] using '{label}' for stratified sampling and train/test split")
        sample_df = stratified_draw(df, by=label, n_total=min(args.sample_size, len(df)), seed=args.seed)
    else:
        print("[label] no suitable label found; falling back to random sampling")
        sample_df = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed)

    sample_df.to_csv(args.out_sample, index=False)
    print(f"[save] sample -> {args.out_sample} (rows={len(sample_df)})")

    # Train/test split (80/20), stratified if label available and class count >= 2
    if train_test_split and label and sample_df[label].nunique(dropna=True) >= 2:
        train_df, test_df = train_test_split(
            sample_df, test_size=0.2, random_state=args.seed, stratify=sample_df[label]
        )
    else:
        train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=args.seed) if train_test_split \
                            else (sample_df.sample(frac=0.8, random_state=args.seed),
                                  sample_df.drop(sample_df.sample(frac=0.8, random_state=args.seed).index))

    train_df.to_csv(args.out_train, index=False)
    test_df.to_csv(args.out_test, index=False)
    print(f"[save] train -> {args.out_train} (rows={len(train_df)}), test -> {args.out_test} (rows={len(test_df)})")

if __name__ == "__main__":
    main()
