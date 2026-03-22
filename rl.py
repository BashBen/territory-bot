import torch
import torch.optim as optim

from model import TerritoryModel

from game.core import Game

# ----- Helpers ------
def init_hidden(hidden_size, num_layers):
    return (torch.zeros(num_layers, hidden_size), torch.zeros(num_layers, hidden_size))

game = Game(seed=42)
player_id = game.add_player()


lr = 1e-3
gamma = 0.995

model = TerritoryModel()
optimizer = optim.Adam(model.parameters(), lr=lr)

