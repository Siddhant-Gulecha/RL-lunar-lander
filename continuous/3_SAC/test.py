import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------
# Config
# ----------------------------
ENV_NAME = "LunarLanderContinuous-v3"
MODEL_PATH = "sac_actor.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ----------------------------
# Actor Network (MUST MATCH TRAINING SCRIPT)
# ----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_l = nn.Linear(256, action_dim)
        self.log_std_l = nn.Linear(256, action_dim)  # Not used in test, but needed for loading state_dict
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # For testing, we only care about the mean (deterministic action)
        mean = self.mean_l(x)
        # Squash the mean action
        action = torch.tanh(mean) * self.max_action
        return action


# ----------------------------
# Test Loop
# ----------------------------
def test():
    env = gym.make(ENV_NAME, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
    try:
        actor.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded actor from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Train first.")
        return
    except RuntimeError as e:
        print(f"Error loading model. Architecture mismatch? {e}")
        return

    actor.eval()

    for ep in range(1, 6):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                # Get deterministic action (the mean)
                action = actor(state_tensor).cpu().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
        print(f"Episode {ep}: Total Reward = {total_reward:.2f} | Steps: {steps}")

    env.close()


if __name__ == "__main__":
    test()