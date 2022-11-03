import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Discriminator(nn.Module):
    """Discriminator using casual dilated convolution, outputs a probability for each time step
    Args:
        input_size (int): dimensionality (channels) of the input
        n_layers (int): number of hidden layers
        n_channels (int): number of channels in the hidden layers (it's always the same)
        kernel_size (int): kernel size in all the layers
        dropout: (float in [0-1]): dropout rate

    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, 1)
    """

    def __init__(self, input_size, n_layers, n_channel, kernel_size, seq_len, dropout=0):
        super().__init__()
        # Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers
        self.tcn = TCN(input_size, 1, num_channels, kernel_size, seq_len, dropout)

    def forward(self, x, c, channel_last=True):
        z = torch.cat([x, c], dim=2)
        z, features = self.tcn(z, channel_last)

        return torch.sigmoid(z), features


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, seq_len, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1] * seq_len, output_size)
        self.flatten = nn.Flatten()
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, channel_last=True):
        # If channel_last, the expected format is (batch_size, seq_len, features)
        y1, features = self.tcn(x.transpose(1, 2) if channel_last else x)
        self.tcn.features.clear()

        return self.linear(self.flatten(y1.transpose(1, 2))), features


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.features = []
        self.feature_layer = FeatureExtractLayer(self.features)
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout), self.feature_layer]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x), self.features.copy()


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1,
                                 self.conv2, self.chomp2, self.relu2,
                                 self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class FeatureExtractLayer(nn.Module):
    def __init__(self, out_list):
        super(FeatureExtractLayer, self).__init__()
        self.out_list = out_list

    def forward(self, x):
        self.out_list.append(x)
        return x
