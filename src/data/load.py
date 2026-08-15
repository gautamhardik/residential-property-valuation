"""Data-loading helpers for the King County housing dataset.

Responsibilities:
- Load train / test Excel files.
- Deduplicate property ids.
- Attach derived calendar columns (quarter, sale_year) from the `date` string.
- Build a canonical property-id split used by every experiment.
"""
import json

import pandas as pd

from src.config import DATA_TRAIN, DATA_TEST, RANDOM_STATE, TEST_SIZE, PREPROCESSED_DIR


def parse_sale_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the `date` column of the form '20140916T000000' into year/quarter."""
    if "date" not in df.columns:
        return df
    dt = pd.to_datetime(df["date"].str.slice(0, 8), format="%Y%m%d", errors="coerce")
    df = df.copy()
    df["sale_year"] = dt.dt.year
    df["sale_quarter"] = dt.dt.quarter
    return df


def load_train() -> pd.DataFrame:
    df = pd.read_excel(DATA_TRAIN)
    return parse_sale_date(df)


def load_test() -> pd.DataFrame:
    df = pd.read_excel(DATA_TEST)
    return parse_sale_date(df)


def load_clean_train() -> pd.DataFrame:
    """Load the cleaned (deduplicated, parsed-date) training frame cached in Phase 1."""
    return pd.read_pickle(PREPROCESSED_DIR / "train_clean.pkl")


def load_clean_test() -> pd.DataFrame:
    return pd.read_pickle(PREPROCESSED_DIR / "test_clean.pkl")


def canonical_split(df: pd.DataFrame, force: bool = False):
    """Return (train_ids, val_ids) saved on disk so all experiments share one split.

    The split is deterministic and only ever derived from the cleaned *training*
    frame; the test set is never used for selection.
    """
    out_path = PREPROCESSED_DIR / "split.json"
    if out_path.exists() and not force:
        with out_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data["train_ids"]), set(data["val_ids"])
    return _build_split(df, out_path)


def _build_split(df, out_path):
    ids = df["id"].tolist()
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"train_ids": tr, "val_ids": va}, f)
    return set(tr), set(va)


def split_frame(df: pd.DataFrame):
    tr_ids, va_ids = canonical_split(df)
    return df[df["id"].isin(tr_ids)].copy(), df[df["id"].isin(va_ids)].copy()