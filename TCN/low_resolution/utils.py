import torch
import pandas as pd
import numpy as np

class StandardScaler():
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (X-self.mean)/self.std

    def inverse_transform(self, X):
        return X*self.std + self.mean

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, seq_len=1, y_offset=1):
        self.X = X
        self.seq_len = seq_len
        self.y_offset = y_offset

    def __len__(self):
        return self.X.__len__() - (self.seq_len + self.y_offset-1)

    def __getitem__(self, index):
        return self.X[index:index + self.seq_len], self.X[index + self.seq_len + self.y_offset - 1]


def get_traffic_data(train_test_ratio=0.8):
    url_data = 'https://zenodo.org/record/4264005/files/PEMS-BAY.csv'

    df_raw = pd.read_csv(url_data, index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    train_df = df_raw.iloc[:int(train_test_ratio * df_raw.shape[0])]
    test_df = df_raw.iloc[int(train_test_ratio * df_raw.shape[0]):]

    return train_df, test_df
