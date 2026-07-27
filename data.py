import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler

SEQ_LEN     = 30
FEATURE     = "close"
BATCH_SIZE  = 64
TRAIN_RATIO = 0.8
TICKER      = "AAPL"


def load_data(csv_path: str, ticker: str = TICKER):
    df = pd.read_csv(csv_path)
    df = df[df["Name"] == ticker][["date", FEATURE]].dropna()
    df = df.sort_values("date").reset_index(drop=True)
    return df[FEATURE].values.reshape(-1, 1)


def _make_sequences(data: np.ndarray):
    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i : i + SEQ_LEN])
        y.append(data[i + SEQ_LEN])
    return np.array(X), np.array(y)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(csv_path: str, train_ratio: float = TRAIN_RATIO):
    df = pd.read_csv(csv_path)
    tickers = df["Name"].unique()

    train_datasets, test_datasets = [], []
    skipped = 0

    for ticker in tickers:
        prices = df[df["Name"] == ticker][FEATURE].dropna().values.reshape(-1, 1)
        if len(prices) < SEQ_LEN + 2:
            skipped += 1
            continue

        split = int(len(prices) * train_ratio)
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(prices[:split])
        test_scaled  = scaler.transform(prices[split:])

        X_tr, y_tr = _make_sequences(train_scaled)
        X_te, y_te = _make_sequences(test_scaled)

        if len(X_tr) > 0:
            train_datasets.append(StockDataset(X_tr, y_tr))
        if len(X_te) > 0:
            test_datasets.append(StockDataset(X_te, y_te))

    print(f"Loaded {len(tickers) - skipped} tickers  |  skipped {skipped} (too short)")

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(ConcatDataset(test_datasets),  batch_size=BATCH_SIZE)
    return train_loader, test_loader
