import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


ENV_NAME = "LunarLander-v3"
MODEL_PATH = "double_dqn_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test():
    env = gym.make(ENV_NAME, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(DEVICE)

    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
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
                action = policy_net(state_t).argmax(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
        print(f"Episode {ep}: Total Reward = {total_reward:.2f} | Steps: {steps}")

    env.close()


if __name__ == "__main__":
    test()