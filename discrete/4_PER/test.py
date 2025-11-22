import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ----------------------------
# Dueling DQN Model (MUST MATCH TRAINING SCRIPT)
# ----------------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_dim=8, output_dim=4):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals


# ----------------------------
# Config
# ----------------------------
ENV_NAME = "LunarLander-v3"
MODEL_PATH = "per_dqn_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Test Loop
# ----------------------------
def test():
    env = gym.make(ENV_NAME, render_mode="human")
    policy_net = DuelingDQN().to(DEVICE)

    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find model file '{MODEL_PATH}'.")
        return

    policy_net.eval()

    for ep in range(1, 6):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = policy_net(state_t).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
        print(f"Episode {ep}: Total Reward = {total_reward:.2f} | Steps: {steps}")

    env.close()


if __name__ == "__main__":
    test()