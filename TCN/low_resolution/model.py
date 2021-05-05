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
        self.tcn = TemporalConvNet(output_size+2, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.euler_clock = TimePassing(dt)
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: (batch x seq_length x (features + time features (2))
        # y: (batch x 1 x time features (2))
        y = self.euler_clock(x.transpose(1, 2)).transpose(1, 2)  # wind a tick (anti-clockwise)
        # x:
        # x = x[:, :self.selfoutput_size, :]
        # x = self.tcn(x_t.transpose(1, 2)).transpose(1, 2)
        x = self.tcn(x)
        # x = self.linear(torch.cat((x[:, -1, :], torch.squeeze(y)), dim=-1)).unsqueeze(1)
        x = self.linear(x[:, :, -1])
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
