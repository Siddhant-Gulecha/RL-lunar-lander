import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import os
from collections import deque
from gymnasium.wrappers import TimeLimit

# ----------------------------
# Config
# ----------------------------
ENV_NAME = "LunarLanderContinuous-v3"
EPISODES = 2000
LR = 3e-4  # Same learning rate for all networks
GAMMA = 0.99
TAU = 0.005  # For soft target updates
BATCH_SIZE = 256
REPLAY_SIZE = 100000
WARMUP = 5000  # Timesteps of random actions before training
MAX_EPISODE_STEPS = 500  # Enforce a time limit
AUTO_ALPHA = True  # Automatically tune the temperature (alpha)
ALPHA = 0.2  # Fixed alpha value if AUTO_ALPHA=False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(action), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(reward), dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(np.array(next_state), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(done), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------------
# Networks
# ----------------------------
class Actor(nn.Module):
    """
    SAC Policy Network (Stochastic)
    Outputs parameters for a squashed Gaussian distribution.
    """

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_l = nn.Linear(256, action_dim)
        self.log_std_l = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean_l(x)
        log_std = self.log_std_l(x)

        # Constrain log_std for stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Sample using reparameterization trick
        x_t = dist.rsample()

        # Squash action to be between -1 and 1 (using tanh)
        y_t = torch.tanh(x_t)

        # Action sent to environment
        action = y_t * self.max_action

        # Calculate log probability (with correction for tanh squashing)
        log_prob = dist.log_prob(x_t)
        log_prob = log_prob - torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """
    SAC Twin Q-Network (Critic)
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Critic 2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


# ----------------------------
# Training Loop
# ----------------------------
def train():
    env = gym.make(ENV_NAME)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR)

    critic = Critic(state_dim, action_dim).to(DEVICE)
    critic_target = Critic(state_dim, action_dim).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR)

    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    # Alpha (temperature) setup
    if AUTO_ALPHA:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(DEVICE)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        alpha_optimizer = optim.Adam([log_alpha], lr=LR)
        alpha = log_alpha.exp()
    else:
        alpha = ALPHA
        target_entropy = None  # Not needed
        log_alpha = None
        alpha_optimizer = None

    training_scores = []
    best_score = -np.inf
    total_timesteps = 0

    print(f"Starting SAC training on {DEVICE}...")

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            total_timesteps += 1

            if total_timesteps < WARMUP:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action, _ = actor.sample(state_tensor)
                    action = action.cpu().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= BATCH_SIZE and total_timesteps > WARMUP:
                state_b, action_b, reward_b, next_state_b, done_b = replay_buffer.sample(BATCH_SIZE)

                # --- Critic Update ---
                with torch.no_grad():
                    # Get next action and log prob
                    next_actions, next_log_probs = actor.sample(next_state_b)

                    # Get target Q-values from twin critics
                    q1_target, q2_target = critic_target(next_state_b, next_actions)
                    q_target_min = torch.min(q1_target, q2_target)

                    # Add entropy term to target
                    q_target = q_target_min - alpha * next_log_probs

                    # Calculate Bellman target
                    y = reward_b + (1 - done_b) * GAMMA * q_target

                q1_current, q2_current = critic(state_b, action_b)
                critic_loss = F.mse_loss(q1_current, y) + F.mse_loss(q2_current, y)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # --- Actor Update ---
                actions_pi, log_probs_pi = actor.sample(state_b)
                q1_pi, q2_pi = critic(state_b, actions_pi)
                q_pi_min = torch.min(q1_pi, q2_pi)

                actor_loss = (alpha * log_probs_pi - q_pi_min).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # --- Alpha Update (if auto-tuning) ---
                if AUTO_ALPHA:
                    alpha_loss = -(log_alpha * (log_probs_pi + target_entropy).detach()).mean()

                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    alpha = log_alpha.exp()

                # --- Soft Target Updates ---
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        training_scores.append(episode_reward)

        if ep % 20 == 0:
            avg_score = np.mean(training_scores[-20:])
            print(
                f"Ep {ep:4d} | Timesteps: {total_timesteps:7d} | Avg Score: {avg_score:7.2f} | Alpha: {alpha.item():.3f}")
            if avg_score > best_score:
                best_score = avg_score
                torch.save(actor.state_dict(), "sac_actor.pth")

    np.save("sac_scores.npy", np.array(training_scores))
    env.close()


if __name__ == "__main__":
    train()