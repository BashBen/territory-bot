import torch
import torch.optim as optim
import numpy as np
import logging

from model import TerritoryModel

from game.core import Game

# ----- Model Params -----
lr = 1e-3
GAMMA = 0.995
input_size = 128
hidden_size = 128
num_layers = 2
output_size = 4
input_channels = 2

# ----- Logger -----
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----- Helpers ------
def init_hidden(hidden_size, num_layers):
    return (torch.zeros(num_layers, 1, hidden_size), torch.zeros(num_layers, 1, hidden_size))

def get_reward(init_map, final_map):
    size = len(init_map)
    init_unique, init_count = np.unique(init_map, return_counts=True)
    final_unique, final_count = np.unique(final_map, return_counts=True)

    init_count = dict(zip(init_unique, init_count)).get(2)
    final_count = dict(zip(final_unique, final_count)).get(2)

    reward = (final_count - init_count) / size
    return reward


def collect_episode(game: Game, model: TerritoryModel):
    player_id = game.add_player()
    map_x_size = game.get_state(relative=player_id).shape[1]
    map_y_size = game.get_state(relative=player_id).shape[2]

    hidden = init_hidden(hidden_size, num_layers)

    log_probs = []
    rewards = []

    # temp secondary map
    # balance_map = np.zeros(512, 512)

    done = False # Don't need done, since we're checking for game_over event and returning values
    tick_count = 0

    while not done:
        map_state = game.get_state()

        model_input = torch.from_numpy(map_state).unsqueeze(0)

        mu, std, action_logits, hidden = model(model_input.float(), hidden)
        hidden = tuple(h.detach() for h in hidden)

        dist = torch.distributions.Normal(mu, std) 
        # sample = dist.sample()

        x, y, strength = dist.sample().squeeze(0) # Should return a 1d tensor with a sample from each dist

        adjusted_coords = torch.clamp(torch.stack([x, y]), 0, 511)
        adjusted_x, adjusted_y = torch.round(adjusted_coords)
        adjusted_strength = torch.clamp(strength, 0, 1)

        # log_probs.append(dist.log_prob(torch.stack([adjusted_x, adjusted_y, adjusted_strength])).sum())

        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()

        log_probs.append(action_dist.log_prob(action) + dist.log_prob(torch.stack([adjusted_x, adjusted_y, adjusted_strength])).sum()) # add action log prob to total

        if action.item() == 1:
            action_payload = {
                "type": "attack",
                "target": [adjusted_x.item(), adjusted_y.item()],
                "percentage": adjusted_strength.item()
            }
            # Apply penalty if input is invalid
            action_submitted = game.action(player_id, payload=action_payload)

            if not action_submitted:
                logging.info(f'Input from tick {tick_count} was invalid: {action_payload}')

        events = game.tick()
        next_map = game.get_state()[0]

        reward = get_reward(map_state[0], next_map)
        rewards.append(reward)

        for event in events:
            if (event.type == "game_won") or (event.type == "player_game_over" and event.player_id == player_id):
                done = True
                return log_probs, rewards, event.type
        
        if tick_count >= 10000:
            return log_probs, rewards, event.type
            
        tick_count += 1


def calculate_loss(log_probs: list, rewards: list):
    returns = []
    G = 0

    for reward in reversed(rewards):
        G = reward + GAMMA * G
        returns.insert(0, G)

    returns = torch.tensor(returns)

    loss = 0

    for log_prob, G_t in zip(log_probs, returns):
        loss += -log_prob * G_t

    return loss


# ----- Main Loop -----
model = TerritoryModel(input_size, input_channels, hidden_size, num_layers)
optimizer = optim.Adam(model.parameters(), lr=lr)


for i in range(1):
    game = Game()

    log_probs, rewards, result = collect_episode(game, model)
    print(result)
    print("Game done")

    loss = calculate_loss(log_probs, rewards)
    print(f"Episode {i}: \nrewards are: {rewards}\nloss is {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
