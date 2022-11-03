import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_size, hidden=256, out_size=1, num_layers=1):
        super(Generator, self).__init__()
        self.n_layers = num_layers
        self.hidden_dim = hidden
        self.out_dim = out_size

        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden, out_size), nn.Tanh())

    def forward(self, z, c, device):
        x = torch.cat([z, c], dim=2)
        batch_size, seq_len = x.size(0), x.size(1)

        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

        recurrent_features, _ = self.lstm(x, (h_0, c_0))
        outputs = self.linear(recurrent_features)
        return outputs
