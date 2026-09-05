import os

import numpy as np
import torch

from .nutrition_env import NutritionEnvironment
from .dqn_agent import NutritionDQNAgent, state_to_vector


# ============================================================
# CONFIGURATION
# ============================================================

EPISODES = 500

MODEL_DIR = os.path.dirname(__file__)


# ============================================================
# TRAIN ONE GOAL
# ============================================================

def train_goal(goal):

    print()
    print("=" * 60)
    print(f"TRAINING RL AGENT: {goal.upper()}")
    print("=" * 60)

    env = NutritionEnvironment()

    agent = NutritionDQNAgent(
        state_size=7,
        action_size=len(env.actions),
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    )

    rewards_history = []
    losses_history = []

    for episode in range(1, EPISODES + 1):

        state = env.reset(
            goal=goal
        )

        state_vector = state_to_vector(
            state
        )

        total_reward = 0.0

        # ----------------------------------------------------
        # Multiple interactions per episode
        # ----------------------------------------------------

        for step in range(10):

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

            loss = agent.train_step(
                batch_size=32
            )

            if loss is not None:
                losses_history.append(
                    loss
                )

            state_vector = next_state_vector

            total_reward += reward

            if done:
                break

        rewards_history.append(
            total_reward
        )

        # ----------------------------------------------------
        # Training progress
        # ----------------------------------------------------

        if episode % 50 == 0:

            recent_rewards = rewards_history[
                -50:
            ]

            average_reward = np.mean(
                recent_rewards
            )

            print(
                f"Episode {episode:03d} | "
                f"Avg Reward: {average_reward:.4f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    goal_filename = {
        "Weight Loss": "nutritwin_rl_weight_loss.pth",
        "Weight Gain": "nutritwin_rl_weight_gain.pth",
        "Maintain Weight": "nutritwin_rl_maintenance.pth"
    }[goal]

    model_path = os.path.join(
        MODEL_DIR,
        goal_filename
    )

    torch.save(
        {
            "goal": goal,
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "actions": env.actions,
            "model_state_dict": agent.q_network.state_dict(),
            "epsilon": agent.epsilon,
            "gamma": agent.gamma
        },
        model_path
    )

    print()
    print(
        f"✅ Saved RL model:"
    )

    print(
        model_path
    )

    print(
        f"Final Average Reward: "
        f"{np.mean(rewards_history[-50:]):.4f}"
    )

    return agent


# ============================================================
# MAIN TRAINING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NutriTwinNet Multi-Goal Reinforcement Learning")
    print("=" * 60)

    print(
        f"Episodes per goal: {EPISODES}"
    )

    train_goal(
        "Weight Loss"
    )

    train_goal(
        "Weight Gain"
    )

    train_goal(
        "Maintain Weight"
    )

    print()
    print("=" * 60)
    print("✅ ALL THREE RL AGENTS TRAINED SUCCESSFULLY")
    print("=" * 60)

    print()
    print(
        "rl/nutritwin_rl_weight_loss.pth"
    )

    print(
        "rl/nutritwin_rl_weight_gain.pth"
    )

    print(
        "rl/nutritwin_rl_maintenance.pth"
    )