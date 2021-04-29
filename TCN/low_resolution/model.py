from torch import nn
from TCN.tcn import TemporalConvNet


class LowResolutionTCN(nn.Module):
    def __init__(self, input_size: int,
                 seq_length: int,
                 num_channels: list,
                 kernel_size: int,
                 dropout: float):
        super(LowResolutionTCN, self).__init__()
        self.tcn = TemporalConvNet(seq_length, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], input_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        return x
