from torch import nn
from TCN.tcn import TemporalConvNet


class Encoder(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
        return self.linear(x)


class Decoder(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
        return self.linear(x)


class LowResolutionTCN(nn.Module):
    def __init__(self, input_size: int,
                 compress_dim: int,
                 seq_length: int,
                 num_channels: list,
                 kernel_size: int,
                 dropout: float):
        super(LowResolutionTCN, self).__init__()
        self.encoder = Encoder(input_size, compress_dim)
        self.decoder = Decoder(compress_dim, input_size)
        self.tcn = TemporalConvNet(seq_length, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], compress_dim)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.encoder(x)
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        x = self.decoder(x)
        return x
