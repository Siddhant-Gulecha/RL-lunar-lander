import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# ----------------------------
# Config
# ----------------------------
ENV_NAME = "LunarLanderContinuous-v3"
MODEL_PATH = "ppo_actor_critic_best.pth"
STATS_PATH = "ppo_norm_stats.pth"  # Path to saved normalization stats
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


# ----------------------------
# Actor-Critic Network (Must match training)
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCritic, self).__init__()
        self.max_action = max_action
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.critic_head = nn.Linear(256, 1)
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_deterministic_action(self, state):
        with torch.no_grad():
            x = self.shared_net(state)
            mean = self.actor_mean(x)
            action = torch.tanh(mean) * self.max_action
        return action.cpu().numpy()


# ----------------------------
# Normalization Helper
# ----------------------------
class NormalizeObservation:
    def __init__(self, mean, var, clip=10.0):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=DEVICE)
        self.var = torch.tensor(var, dtype=torch.float32, device=DEVICE)
        self.clip = clip
        self.epsilon = 1e-8

    def __call__(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        # Apply normalization
        obs_norm = (obs_tensor - self.mean) / (torch.sqrt(self.var) + self.epsilon)
        # Clip the normalized observations
        obs_norm = torch.clamp(obs_norm, -self.clip, self.clip)
        return obs_norm.unsqueeze(0)  # Add batch dim


# ----------------------------
# Test Loop
# ----------------------------
def test():
    env = gym.make(ENV_NAME, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --- Load Normalization Stats ---
    try:
        mean, var = torch.load(STATS_PATH, map_location=DEVICE, weights_only=False)
        normalizer = NormalizeObservation(mean, var)
        print(f"Loaded normalization stats from {STATS_PATH}")
    except FileNotFoundError:
        print(f"Error: {STATS_PATH} not found. Train first.")
        return

    # --- Load Agent ---
    agent = ActorCritic(state_dim, action_dim, max_action).to(DEVICE)
    try:
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Train first.")
        return

    agent.eval()

    for ep in range(1, 6):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            # --- CRITICAL: Normalize the state ---
            state_tensor = normalizer(state)

            action = agent.get_deterministic_action(state_tensor)[0]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
        print(f"Episode {ep}: Total Reward = {total_reward:.2f} | Steps: {steps}")

    env.close()


if __name__ == "__main__":
    test()