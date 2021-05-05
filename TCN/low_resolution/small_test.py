from TCN.low_resolution.model import LowResolutionTCN
import torch
from torch import nn
import pandas as pd
from TCN.low_resolution.utils import TimeSeriesDataset, StandardScaler, get_euler_time
import numpy as np

df_0 = pd.read_csv('/home/semin_noadmin/workspace/TCN/low_resol.csv', index_col=0).fillna(0)
df_0.index = pd.to_datetime(df_0.index)

df_train, df_valid = df_0[:'2017-05-15'], df_0['2017-05-16':]

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
model = LowResolutionTCN(output_size=3,
                         seq_length=10,
                         num_channels=[5] * 5,
                         kernel_size=5,
                         dropout=0.1,
                         dt=df_train.index[1] - df_train.index[0])
device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)e
model.to(device)

scaler = StandardScaler(mask=(0, 3))
X_train = scaler.fit_transform(np.column_stack((df_train.values, get_euler_time(df_train.index))))
X_valid = scaler.transform(np.column_stack((df_valid.values, get_euler_time(df_valid.index))))

train_dataset = TimeSeriesDataset(X_train, seq_len=10)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=8)
steps_ahead = 3
for (x, y) in train_loader:
    with torch.no_grad():
        # x: (batch x seq_length x (features + time features (2))
        # y: (batch x 1 x (features + time features (2))
        x, y = x.float().to(device), y.float().to(device).unsqueeze(1)
        for j in range(steps_ahead):
            output = model(x)
            x = torch.cat((x[:, 1:, :], output), dim=1)

    output = model(x)


