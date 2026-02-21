import pandas as pd

def clean_numeric_columns(df):
    numeric_cols = ["Price", "Open", "High", "Low"]

    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = df[col].astype(float)

    if "Vol." in df.columns:
        df["Vol."] = df["Vol."].astype(str).str.replace(",", "", regex=False)
        df["Vol."] = df["Vol."].str.replace("K", "", regex=False)
        df["Vol."] = df["Vol."].str.replace("M", "", regex=False)
        df["Vol."] = df["Vol."].astype(float)

    return df


def create_features(df):
    df["Prev_Close"] = df["Price"].shift(1)
    df.dropna(inplace=True)
    return df
