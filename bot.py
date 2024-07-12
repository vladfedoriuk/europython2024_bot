# SPDX-License-Identifier: BSD-3-Clause
import itertools
import random
# flake8: noqa F401
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from vendeeglobe import (
    Checkpoint,
    Heading,
    Instructions,
    Location,
    Vector,
    config,
)
from vendeeglobe.utils import distance_on_surface

import math
import random
from collections import namedtuple, deque
from torch.nn import functional as F


import torch
from torch import nn, optim

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
# actions:
#: left 15 degrees
#: right 15 degrees
#: left 30 degrees
#: right 30 degrees
#: left 45 degrees
#: right 45 degrees
#: go to location
n_actions = 7
# Get the number of state observations
map_size = 100
n_observations = 5 + map_size**2

# if GPU is to be used
device = torch.device("cpu")

weights_path = Path(__file__).parent


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(1024)
        self.layer2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.layer_norm2 = nn.LayerNorm(512)
        self.layer3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.layer_norm3 = nn.LayerNorm(256)
        self.layer4 = nn.Linear(256, n_actions)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.layer_norm1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.layer_norm2(x)
        x = F.relu(self.layer3(x))
        x = self.dropout3(x)
        x = self.layer_norm3(x)
        return self.layer4(x)


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# if Path(weights_path / "policy_net.pth").exists():
#     print("Loading weights - policy")
#     policy_net.load_state_dict(torch.load(weights_path / "policy_net.pth"))
# if Path(weights_path / "target_net.pth").exists():
#     print("Loading weights - target")
#     target_net.load_state_dict(torch.load(weights_path / "target_net.pth"))

memory = ReplayMemory(10_000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample <= eps_threshold:
        return torch.tensor(
            [[random.randint(0, n_actions - 1)]], device=device, dtype=torch.long
        )
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return policy_net(state).max(1).indices.view(1, 1)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    if steps_done % 100 == 0:
        # torch.save(policy_net.state_dict(), weights_path / "policy_net.pth")
        # torch.save(target_net.state_dict(), weights_path / "target_net.pth")
        pass


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

def trunc_lat(lat):
    return max(-90, min(90, lat))


def trunc_lon(lon):
    return max(-180, min(180, lon))

class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "The Ifers"  # This is your team name
        # This is the course that the ship has to follow
        self.course = [
            Checkpoint(latitude=43.797109, longitude=-11.264905, radius=200.),
            Checkpoint(longitude=-29.908577, latitude=17.999811, radius=300.),
            Checkpoint(latitude=-11.441808, longitude=-29.660252, radius=200.),
            Checkpoint(longitude=-63.240264, latitude=-61.025125, radius=100.),
            Checkpoint(latitude=2.806318, longitude=-168.943864, radius=1990.0),
            Checkpoint(latitude=-62.052286, longitude=169.214572, radius=600.0),
            Checkpoint(latitude=-15.668984, longitude=77.674694, radius=1190.0),
            Checkpoint(latitude=-39.438937, longitude=19.836265, radius=200.0),
            Checkpoint(latitude=14.881699, longitude=-21.024326, radius=100.0),
            Checkpoint(latitude=44.076538, longitude=-18.292936, radius=100.0),
            Checkpoint(
                latitude=config.start.latitude,
                longitude=config.start.longitude,
                radius=5,
            ),
        ]
        # zeros of n_observations
        self.state = torch.zeros(n_observations, device=device).unsqueeze(0)
        self.previous_location = None
        self.dt_stuck = 0
        self.dt_in_radius = 0

    def run(
        self,
        t: float,
        dt: float,
        longitude: float,
        latitude: float,
        heading: float,
        speed: float,
        vector: np.ndarray,
        forecast: Callable,
        world_map: Callable,
    ) -> Instructions:
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in hours.
        dt:
            The time step in hours.
        longitude:
            The current longitude of the ship.
        latitude:
            The current latitude of the ship.
        heading:
            The current heading of the ship.
        speed:
            The current speed of the ship.
        vector:
            The current heading of the ship, expressed as a vector.
        forecast:
            Method to query the weather forecast for the next 5 days.
            Example:
            current_position_forecast = forecast(
                latitudes=latitude, longitudes=longitude, times=0
            )
        world_map:
            Method to query map of the world: 1 for sea, 0 for land.
            Example:
            current_position_terrain = world_map(
                latitudes=latitude, longitudes=longitude
            )

        Returns
        -------
        instructions:
            A set of instructions for the ship. This can be:
            - a Location to go to
            - a Heading to point to
            - a Vector to follow
            - a number of degrees to turn Left
            - a number of degrees to turn Right

            Optionally, a sail value between 0 and 1 can be set.
        """
        # Initialize the instructions
        instructions = Instructions()
        # TODO: Remove this, it's only for testing =================
        current_wind = forecast(latitudes=latitude, longitudes=longitude, times=0)
        current_position_terrain = world_map(latitudes=latitude, longitudes=longitude)

        # ===========================================================

        # Go through all checkpoints and find the next one to reach
        for ch in self.course:
            # Compute the distance to the checkpoint
            dist = distance_on_surface(
                longitude1=longitude,
                latitude1=latitude,
                longitude2=ch.longitude,
                latitude2=ch.latitude,
            )
            # Consider slowing down if the checkpoint is close
            jump = dt * np.linalg.norm(speed)
            if dist < 2.0 * ch.radius + jump:
                instructions.sail = min(ch.radius / jump, 1)
            else:
                instructions.sail = 1.0
            # Check if the checkpoint has been reached
            if dist < ch.radius:
                ch.reached = True
            if not ch.reached:
                # actions:
                #: left 15 degrees
                #: right 15 degrees
                #: left 30 degrees
                #: right 30 degrees
                #: left 45 degrees
                #: right 45 degrees
                #: go to location
                action = select_action(self.state)
                action_item = action.item()

                if action_item == 0:
                    instructions.left = 5
                elif action_item == 1:
                    instructions.right = 5
                elif action_item == 2:
                    instructions.left = 10
                elif action_item == 3:
                    instructions.right = 10
                elif action_item == 4:
                    instructions.left = 15
                elif action_item == 5:
                    instructions.right = 15
                elif action_item == 6:
                    instructions.location = Location(
                        longitude=ch.longitude, latitude=ch.latitude
                    )
                else:
                    raise ValueError("Invalid action")
                map_20_20 = np.zeros((map_size, map_size))
                for i, j in itertools.product(
                    range(-map_size // 2, map_size // 2),
                    range(-map_size // 2, map_size // 2),
                ):
                    try:
                        map_20_20[i + map_size // 2, j + map_size // 2] = world_map(
                            latitudes=trunc_lat(latitude + i/1000),
                            longitudes=trunc_lon(longitude + j/1000),
                        )
                    except:
                        map_20_20[i + map_size // 2, j + map_size // 2] = 0

                    next_state = torch.tensor(
                        [
                            t,
                            longitude,
                            latitude,
                            *current_wind,
                            *map_20_20.flatten(),
                        ],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)

                    # compute reward as the negative of the distance to the checkpoint and the positive of the speed
                    # we reward the decrease in distance and the increase in speed
                    # we penalize the time spent
                    if self.previous_location is None:
                        dist_from_prev_loc_to_loc = 1
                        dist_from_prev_loc = 1
                    else:
                        dist_from_prev_loc_to_loc = distance_on_surface(
                            longitude1=self.previous_location.longitude,
                            latitude1=self.previous_location.latitude,
                            longitude2=longitude,
                            latitude2=latitude,
                        )
                        dist_from_prev_loc = distance_on_surface(
                            longitude1=self.previous_location.longitude,
                            latitude1=self.previous_location.latitude,
                            longitude2=ch.longitude,
                            latitude2=ch.latitude,
                        )
                    reward = (100 * (speed + dist_from_prev_loc_to_loc / dt) / 2 ) / (100 * (dist_from_prev_loc - dist) + 10 * t + 1)
                    if dist < ch.radius:
                        self.dt_in_radius += dt
                        if self.dt_in_radius > 1:
                            reward -= self.dt_in_radius / 4
                    else:
                        self.dt_in_radius = 0
                    print("Speed", speed)
                    print("Dist from prev loc to loc", dist_from_prev_loc_to_loc)
                    if abs(dist_from_prev_loc_to_loc) < 0.01 and speed < 0.01:
                        print("Stuck")
                        self.dt_stuck += dt
                        if self.dt_stuck > 1:
                            print("Random direction")
                            instructions.left = None
                            instructions.heading = None
                            instructions.sail = None
                            instructions.right = None
                            random_vector = np.random.rand(2).tolist()
                            instructions.vector = Vector(u=random_vector[0], v=random_vector[1])
                    else:
                        self.dt_stuck = 0
                    reward -= self.dt_stuck / 2
                    self.previous_location = Location(longitude=longitude, latitude=latitude)
                    print(reward)
                    reward = torch.tensor([reward], device=device, dtype=torch.float32)
                    # Store the transition in memory
                    memory.push(self.state, action, next_state, reward)

                    # Move to the next state
                    self.state = next_state

                    # Perform one step of the optimization (on the policy network)
                    optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                    1 - TAU)
                    target_net.load_state_dict(target_net_state_dict)
                    break
                break
        return instructions
