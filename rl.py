import torch
import torch.optim as optim

from model import TerritoryModel

lr = 1e-3
gamma = 0.995

model = TerritoryModel()
lr = lr
gamma = gamma # Discount rate - how much later rewards count now: 0 <= gamma <= 1
optimizer = optim.Adam(model.parameters(), lr=lr)