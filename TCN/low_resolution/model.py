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
        self.tcn = TemporalConvNet(seq_length, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.euler_clock = TimePassing(dt)
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.euler_clock(x)
        x = self.tcn(x)
        x = self.linear(x[:, :, -1]).unsqueeze(1)
        x = torch.cat((x, y), dim=-1)
        return x


class TimePassing(nn.Module):
    def __init__(self, dt) -> object:
        super(TimePassing, self).__init__()
        self.min_to_sec = 60
        self.hour_to_min = 60
        self.max_second_per_day = 24 * self.hour_to_min * self.min_to_sec
        self.dt_sec = dt.hour * self.hour_to_min * self.min_to_sec + dt.minute * self.min_to_sec + dt.second

    def forward(self, x):
        x = x[:, -1, :].copy()  # only need the last element (x_t)
        cos, sin = x[:, :, -2], x[:, :, -1]
        new_cos = cos * torch.cos(self.dt_sec / self.max_second_per_day * 2 * np.pi) - \
                  sin * torch.sin(self.dt_sec / self.max_second_per_day * 2 * np.pi)
        new_sin = cos * torch.sin(self.dt_sec / self.max_second_per_day * 2 * np.pi) + \
                  sin * torch.cos(self.dt_sec / self.max_second_per_day * 2 * np.pi)
        x[:, :, -2] = new_cos
        x[:, :, -1] = new_sin
        return x
