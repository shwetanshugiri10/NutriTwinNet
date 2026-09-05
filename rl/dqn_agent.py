import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .nutrition_env import NutritionEnvironment


class NutritionQNetwork(nn.Module):

    def __init__(
        self,
        state_size,
        action_size
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )


    def forward(self, state):

        return self.network(state)


class NutritionDQNAgent:

    def __init__(
        self,
        state_size=7,
        action_size=10,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(
            maxlen=5000
        )

        self.q_network = NutritionQNetwork(
            state_size,
            action_size
        )

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()


    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


    def act(self, state):

        if random.random() < self.epsilon:

            return random.randrange(
                self.action_size
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.q_network(
                state_tensor
            )

        return int(
            torch.argmax(q_values)
        )


    def train_step(
        self,
        batch_size=32
    ):

        if len(self.memory) < batch_size:

            return None

        batch = random.sample(
            self.memory,
            batch_size
        )

        states = torch.tensor(
            np.array(
                [item[0] for item in batch]
            ),
            dtype=torch.float32
        )

        actions = torch.tensor(
            [item[1] for item in batch],
            dtype=torch.long
        )

        rewards = torch.tensor(
            [item[2] for item in batch],
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.array(
                [item[3] for item in batch]
            ),
            dtype=torch.float32
        )

        dones = torch.tensor(
            [item[4] for item in batch],
            dtype=torch.float32
        )

        current_q = self.q_network(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():

            next_q = self.q_network(
                next_states
            ).max(
                dim=1
            )[0]

        target_q = (
            rewards
            +
            self.gamma
            * next_q
            * (1 - dones)
        )

        loss = self.loss_function(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

        return float(loss.item())


def state_to_vector(state):

    return np.array(
        [
            state["age"] / 100,
            state["weight"] / 200,
            state["height"] / 250,
            state["steps"] / 20000,
            state["heart_rate"] / 200,
            state["sleep_hours"] / 12,
            state["active_calories"] / 2000
        ],
        dtype=np.float32
    )


if __name__ == "__main__":

    print("=" * 60)
    print("NutriTwinNet DQN Reinforcement Learning Agent")
    print("=" * 60)

    env = NutritionEnvironment()

    agent = NutritionDQNAgent(
        state_size=7,
        action_size=len(env.actions)
    )

    print()
    print(
        f"Actions: {len(env.actions)}"
    )

    print(
        f"State size: {agent.state_size}"
    )

    print()
    print("Available Actions:")

    for index, action in enumerate(
        env.actions
    ):

        print(
            f"{index:02d}. {action}"
        )

    state = env.reset(
        goal="Weight Gain"
    )

    state_vector = state_to_vector(
        state
    )

    action_index = agent.act(
        state_vector
    )

    action = env.actions[
        action_index
    ]

    next_state, reward, done = env.step(
        action
    )

    next_state_vector = state_to_vector(
        next_state
    )

    agent.remember(
        state_vector,
        action_index,
        reward,
        next_state_vector,
        done
    )

    print()
    print(
        f"Selected Action: {action}"
    )

    print(
        f"Reward: {reward}"
    )

    print(
        f"Memory Size: {len(agent.memory)}"
    )

    print()
    print("=" * 60)
    print(
        "DQN agent created successfully."
    )
    print("=" * 60)