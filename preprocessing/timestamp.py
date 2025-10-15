import pandas as pd
import numpy as np

def time_features(df, ts_col="Timestamp"):
    """Parse and normalize timestamp values (UTC)"""
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df["hour"] = ts.dt.hour
        df["dow"] = ts.dt.dayofweek
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

        df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
        df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
        df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7.0)
        df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7.0)
    else:
        for c in ["hour", "dow", "is_weekend"]:
            if c not in df.columns:
                df[c] = np.nan
    return df