import torch
import torch.optim as optim
import numpy as np
import logging

from model import TerritoryModel

from game.core import Game

# ----- Model Params -----
lr = 1e-3
gamma = 0.995
input_size = (2, 512, 512)
hidden_size = 128
num_layers = 2
output_size = 4

# ----- Logger -----
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----- Helpers ------
def init_hidden(hidden_size, num_layers):
    return (torch.zeros(num_layers, hidden_size), torch.zeros(num_layers, hidden_size))



def collect_episode(model: TerritoryModel):
    game = Game()
    map_x_size = game.get_state().shape(0)
    map_y_size = game.get_state().shape(1)
    player_id = game.add_player()

    hidden = init_hidden(hidden_size, num_layers)

    log_probs = []
    rewards = []

    # temp secondary map
    # balance_map = np.zeros(512, 512)

    done = False

    while not done:
        map_state = game.get_state()
        model_input = torch.from_numpy(map_state)

        mu, std, action_logits, hidden = model(model_input, hidden)

        dist = torch.distributions.Normal(mu, std) 
        x, y, strength = dist.sample() # Should return a 1d tensor with a sample from each dist

        adjusted_coords = torch.clamp(torch.stack([x, y]), 0, 511)
        adjusted_x, adjusted_y = torch.round(adjusted_coords)
        adjusted_strength = torch.clamp(strength, 0, 1)

        log_probs = dist.log_prob(torch.stack([adjusted_x, adjusted_y, adjusted_strength])).sum()

        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()

        log_probs = action_dist.log_prob(action) + log_probs # add action log prob to total

        if action.item() == 0:
            game.tick()
        else:
            action_payload = {
                "type": "attack",
                "target": [adjusted_x.item(), adjusted_y.item()],
                "percentage": adjusted_strength.item()
            }
            # Apply penalty if input is invalid
            action_submitted = game.action(player_id, payload=action_payload)

            if not action_submitted:
                logging.info(f'')









model = TerritoryModel()
optimizer = optim.Adam(model.parameters(), lr=lr)

