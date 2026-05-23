from pathlib import Path

import pandas as pd


def load_raw(path="data/raw/data.csv"):
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Raw dataset not found: {fp}")
    return pd.read_csv(fp)
