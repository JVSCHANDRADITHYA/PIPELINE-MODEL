# core/ingestion.py

import pandas as pd


def stream_csv(path):
    """
    Generator that yields rows one-by-one as dicts.
    """
    df = pd.read_csv(path)

    if "Timestamp_IST" in df.columns:
        df = df.drop(columns=["Timestamp_IST"])

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().fillna(0)

    for _, row in df.iterrows():
        yield row.to_dict()