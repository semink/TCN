import torch
from torch import nn
import numpy as np
from TCN.tcn import TemporalConvNet


class LowResolutionTCN(nn.Module):
    def __init__(self, output_size: int,
                 seq_length: int,
                 num_channels: list,
                 kernel_size: int,
                 dropout: float,
                 dt):
        super(LowResolutionTCN, self).__init__()
        self.tcn = TemporalConvNet(output_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels[-1] + 2, output_size)
        # self.linear2 = nn.Linear(output_size, output_size)
        # self.fc = nn.Sequential(self.linear1, nn.Sigmoid(), self.linear2)
        self.euler_clock = TimePassing(dt)
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.1)
        # self.linear2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # x: (batch x (features + time features (2) x seq_length)
        # y: (batch x time features (2) x 1)
        y = self.euler_clock(x.transpose(1, 2)).transpose(1, 2)  # wind a tick (anti-clockwise)
        x = self.tcn(x[:, :self.output_size, :])
        x = self.linear1(torch.cat((x[:, :, -1:], y), dim=1).squeeze())
        x = torch.cat((x.unsqueeze(-1), y), dim=1)
        return x


class TimePassing(nn.Module):
    def __init__(self, dt) -> object:
        super(TimePassing, self).__init__()
        self.min_to_sec = 60
        self.hour_to_min = 60
        self.max_second_per_day = 24 * self.hour_to_min * self.min_to_sec
        self.dt_sec = dt.total_seconds()

    def forward(self, x):
        x = x[:, -1:, -2:]  # only need the last element (x_t)
        cos, sin = x[:, :, -2], x[:, :, -1]
        new_cos = cos * np.cos(self.dt_sec / self.max_second_per_day * 2 * np.pi) - \
                  sin * np.sin(self.dt_sec / self.max_second_per_day * 2 * np.pi)
        new_sin = cos * np.sin(self.dt_sec / self.max_second_per_day * 2 * np.pi) + \
                  sin * np.cos(self.dt_sec / self.max_second_per_day * 2 * np.pi)
        x[:, :, -2] = new_cos
        x[:, :, -1] = new_sin
        return x
