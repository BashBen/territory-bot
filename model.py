import torch
import torch.nn as nn
import torch.optim as optim


class TerritoryModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super().__init__()

    self.num_layers = num_layers
    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    self.fl = nn.Linear(hidden_size, output_size)

  def forward(self, x, hx = None):
    # x: (batch_size, sequence_length, input_size)
    # hx: (h0, c0), No memory features for now
    if hx is None:
      hx = (torch.zeros(self.num_layers, self.hidden_size), torch.zeros(self.num_layers, self.hidden_size))
    out, hx = self.lstm(x, hx)
    logits = self.fl(out)
    return logits, hx