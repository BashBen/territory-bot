import torch
import torch.nn as nn
import math


class TerritoryModel(nn.Module):
    def __init__(self, input_size: int, input_channels: int, hidden_size: int, num_layers: int):
        """
        input_size: size of LSTM input
        input_channels: number of channels model will take
        hidden_size: number of params in each hidden layer in LSTM
        num_layers: number of hidden layers in LSTM
        output_size: size of model output
        """
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1), # -> 256x256
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> 128x128
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> 64x64
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), # -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), # -> 16x16
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 512, 512)
            n_flat = self.cnn(dummy).shape[1]

        self.fc1 = nn.Linear(n_flat, input_size) #16 * 16 * 128
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.mu_head = nn.Linear(hidden_size, 3) # For mean of output dist: (mu_x, mu_y, mu_strength)
        self.log_std_head = nn.Linear(hidden_size, 3) # For std of output dist: (std_x, std_y, std_strength)
        self.action_head = nn.Linear(hidden_size, 2) # For actions: (no attack, attack)
        
    def forward(self, x: torch.Tensor, hx: tuple):
        # x: (batch_size, channels, 512, 512)
        # hx: (h0, c0), initially no memory
        x = self.cnn(x)
        x = nn.functional.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x, hx = self.lstm(x, hx)
        x = x.squeeze(1)

        mu = self.mu_head(x)
        std = torch.exp(self.log_std_head(x))
        action_logits = nn.functional.softmax(self.action_head(x)) 
        return mu, std, action_logits, hx