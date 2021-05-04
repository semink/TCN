import torch
import pandas as pd
import numpy as np


class StandardScaler():
    def __init__(self, mean=0, std=1, mask=(0, 3)):
        self.mean = mean
        self.std = std
        self.mask = mask

    def fit(self, X):
        self.mean = np.mean(X[:,self.mask[0]:self.mask[1]], axis=0)
        self.std = np.std(X[:,self.mask[0]:self.mask[1]], axis=0)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X[:,self.mask[0]:self.mask[1]] = (X[:,self.mask[0]:self.mask[1]] - self.mean) / self.std
        return X 

    def inverse_transform(self, X):
        X[self.mask[0]:self.mask[1]] = X[self.mask[0]:self.mask[1]] * self.std + self.mean
        return X


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, seq_len=1, y_offset=1):
        self.X = X
        self.seq_len = seq_len
        self.y_offset = y_offset

    def __len__(self):
        return self.X.__len__() - (self.seq_len + self.y_offset - 1)

    def __getitem__(self, index):
        return self.X[index:index + self.seq_len], self.X[index + self.seq_len + self.y_offset - 1]


def get_euler_time(hour_time_vec):
    """
    @param hour_time_vec: pandas datetime vector
    """
    min_to_sec = 60
    hour_to_min = 60
    max_second_per_day = 24 * hour_to_min * min_to_sec

    time_to_float = hour_time_vec.hour * hour_to_min * min_to_sec \
                    + hour_time_vec.minute * min_to_sec + hour_time_vec.second
    return np.column_stack((np.cos(time_to_float / max_second_per_day * 2 * np.pi),
                            np.sin(time_to_float / max_second_per_day * 2 * np.pi)))

def get_traffic_data(train_test_ratio=0.8):
    url_data = 'https://zenodo.org/record/4264005/files/PEMS-BAY.csv'

    df_raw = pd.read_csv(url_data, index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    train_df = df_raw.iloc[:int(train_test_ratio * df_raw.shape[0])]
    test_df = df_raw.iloc[int(train_test_ratio * df_raw.shape[0]):]

    return train_df, test_df
