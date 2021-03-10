import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd


def generate_dataset(seq_size, df):
    X, Y = [], []
    for i in range(df.shape[0] - seq_size - 1):
        X.append(df.iloc[i:i + seq_size].values)
        Y.append(df.iloc[i + seq_size + 1].values)
    return torch.from_numpy(np.asarray(X)), torch.from_numpy(np.asarray(Y))


def load_dataset(seq_size=288, train_test_ratio=0.8):
    url_data = 'https://zenodo.org/record/4264005/files/PEMS-BAY.csv'
    # url_meta = 'https://zenodo.org/record/4264005/files/PEMS-BAY-META.csv'

    df_raw = pd.read_csv(url_data, index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    train_df = df_raw.iloc[:int(train_test_ratio * df_raw.shape[0])]
    test_df = df_raw.iloc[int(train_test_ratio * df_raw.shape[0]):]

    X_train, Y_train = generate_dataset(seq_size, train_df)
    X_test, Y_test = generate_dataset(seq_size, test_df)

    return X_train, Y_train, X_test, Y_test


def data_generator(N, seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    X_num = torch.rand([N, 1, seq_length])
    X_mask = torch.zeros([N, 1, seq_length])
    Y = torch.zeros([N, 1])
    for i in range(N):
        positions = np.random.choice(seq_length, size=2, replace=False)
        X_mask[i, 0, positions[0]] = 1
        X_mask[i, 0, positions[1]] = 1
        Y[i, 0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
    X = torch.cat((X_num, X_mask), dim=1)
    return Variable(X), Variable(Y)
