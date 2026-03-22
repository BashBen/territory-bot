import torch
import torch.nn as nn
import torch.optim as optim
import math


class TerritoryModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.input_channels = 2 # Two grids (territory, and balance)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self._num_features = math.pow(input_size, 2) * 64
        self.lstm = nn.LSTM(self._num_features, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fl = nn.Linear(self.hidden_size, output_size)

    def forward(self, x: torch.Tensor, hx: tuple):
        # x: (batch_size, channels, 512, 512)
        # hx: (h0, c0), initially no memory
        x = self.cnn(x)
        x = x.unsqueeze(1)
        x, hx = self.lstm(x, hx)
        x = x.squeeze(1)
        logits = self.fl(x)
        return logits, hx