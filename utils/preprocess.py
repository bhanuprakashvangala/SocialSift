import pandas as pd


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['text'])
    return df
